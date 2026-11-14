import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medagent.core.exceptions import ProviderError
from medagent.core.models import Message, ToolCall
from medagent.llm.groq_provider import GroqProvider


def test_tracing_disabled_by_default_leaves_env_untouched() -> None:
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    with patch("medagent.llm.groq_provider.AsyncGroq", return_value=MagicMock()):
        GroqProvider(api_key="test-key", model="test-model")
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_tracing_enabled_sets_langsmith_env_vars() -> None:
    try:
        with patch("medagent.llm.groq_provider.AsyncGroq", return_value=MagicMock()):
            GroqProvider(
                api_key="test-key",
                model="test-model",
                tracing_enabled=True,
                langsmith_api_key="ls-key",
                langsmith_project="medagent-test",
            )
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-key"
        assert os.environ["LANGCHAIN_PROJECT"] == "medagent-test"
    finally:
        for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
            os.environ.pop(key, None)


async def test_complete_raises_provider_error_on_timeout() -> None:
    async def _hangs_forever(**_kwargs: object) -> None:
        await asyncio.sleep(10)

    fake_client = MagicMock()
    fake_client.chat.completions.create = _hangs_forever

    with patch("medagent.llm.groq_provider.AsyncGroq", return_value=fake_client):
        provider = GroqProvider(api_key="test-key", model="test-model", timeout_seconds=0.05)

    with pytest.raises(ProviderError, match="timed out"):
        await provider.complete([Message(role="user", content="hi")])


def _fake_groq_response(
    tool_calls: list[SimpleNamespace] | None, content: str | None = ""
) -> object:
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _provider_returning(response: object) -> tuple[GroqProvider, MagicMock]:
    create = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.chat.completions.create = create
    with patch("medagent.llm.groq_provider.AsyncGroq", return_value=fake_client):
        return GroqProvider(api_key="test-key", model="test-model"), create


async def test_complete_parses_tool_calls_from_response() -> None:
    response = _fake_groq_response(
        [_tool_call("call_1", "search_guidelines", '{"query": "metformin ckd"}')]
    )
    provider, _ = _provider_returning(response)

    result = await provider.complete([Message(role="user", content="hi")])

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].tool_name == "search_guidelines"
    assert result.tool_calls[0].arguments == {"query": "metformin ckd"}


async def test_complete_tolerates_malformed_tool_arguments() -> None:
    """Groq sometimes emits invalid JSON for arguments. That must not crash the
    provider -- an empty-args call lets the tool reject it and the model retry."""
    response = _fake_groq_response([_tool_call("call_1", "search_guidelines", "{not json")])
    provider, _ = _provider_returning(response)

    result = await provider.complete([Message(role="user", content="hi")])

    assert result.tool_calls[0].arguments == {}


async def test_complete_without_tool_calls_returns_empty_list() -> None:
    provider, _ = _provider_returning(_fake_groq_response(None, content="just text"))

    result = await provider.complete([Message(role="user", content="hi")])

    assert result.tool_calls == []
    assert result.content == "just text"


async def test_complete_replays_tool_call_protocol_to_groq() -> None:
    """An assistant turn that requested tools, then the role=tool result that
    answers it by id, must reach Groq in the shape it requires -- otherwise the
    tool message is rejected as orphaned."""
    provider, create = _provider_returning(_fake_groq_response(None, content="done"))
    history = [
        Message(role="user", content="question"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", tool_name="search_guidelines", arguments={"q": "x"})],
        ),
        Message(role="tool", name="search_guidelines", tool_call_id="call_1", content="result"),
    ]

    await provider.complete(history)

    sent = create.call_args.kwargs["messages"]
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] is None
    assert sent[1]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_guidelines", "arguments": '{"q": "x"}'},
        }
    ]
    assert sent[2] == {"role": "tool", "content": "result", "tool_call_id": "call_1"}


# --- reliability: classification, retries, breaker ------------------------------------

import groq  # noqa: E402
import httpx  # noqa: E402

from medagent.core.exceptions import ProviderError as _ProviderError  # noqa: E402
from tests.conftest import FakeClock, metric_value  # noqa: E402

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(status: int, headers: dict[str, str] | None = None, message: str = "err"):
    cls = {
        400: groq.BadRequestError,
        404: groq.NotFoundError,
        413: groq.APIStatusError,
        429: groq.RateLimitError,
        500: groq.InternalServerError,
    }[status]
    response = httpx.Response(status, headers=headers or {}, request=_REQUEST)
    return cls(message, response=response, body=None)


class _FakeSDK:
    """Plays a script of outcomes (exceptions are raised, anything else is
    returned) and repeats the last one when the script runs out."""

    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs: object) -> object:
        self.calls += 1
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _provider(sdk: _FakeSDK, **kwargs: object):
    captured: dict[str, object] = {}

    def factory(**ctor_kwargs: object) -> MagicMock:
        captured.update(ctor_kwargs)
        client = MagicMock()
        client.chat.completions.create = sdk.create
        return client

    with patch("medagent.llm.groq_provider.AsyncGroq", side_effect=factory):
        provider = GroqProvider(api_key="test-key", model="test-model", **kwargs)  # type: ignore[arg-type]
    return provider, captured


def _ok_response(prompt: int = 10, completion: int = 5) -> object:
    return _fake_groq_response(None, content="fine")


_HI = [Message(role="user", content="hi")]


def test_the_sdks_own_retries_are_switched_off() -> None:
    """Left on they stacked under ours: up to 9 HTTP calls per LLM call."""
    _, captured = _provider(_FakeSDK(_ok_response()))
    assert captured["max_retries"] == 0


async def test_a_permanent_error_is_attempted_exactly_once(clock: FakeClock) -> None:
    sdk = _FakeSDK(_status_error(404, message="model does not exist"))
    provider, _ = _provider(sdk)
    before = metric_value("medagent_llm_calls_total", provider="groq", outcome="client_error")

    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)

    assert sdk.calls == 1
    assert excinfo.value.reason == "client_error"
    assert excinfo.value.retryable is False
    assert clock.sleeps == []
    after = metric_value("medagent_llm_calls_total", provider="groq", outcome="client_error")
    assert after - before == 1


async def test_a_short_rate_limit_is_waited_out_then_succeeds(clock: FakeClock) -> None:
    sdk = _FakeSDK(_status_error(429, {"retry-after": "2"}), _ok_response())
    provider, _ = _provider(sdk)

    result = await provider.complete(_HI)

    assert result.content == "fine"
    assert sdk.calls == 2
    assert 2.0 <= clock.sleeps[0] <= 2.5


async def test_a_long_rate_limit_fails_fast_instead_of_holding_the_request(
    clock: FakeClock,
) -> None:
    """The daily-cap 429 that cost 19-91s per request before: retry-after 8m."""
    sdk = _FakeSDK(_status_error(429, {"retry-after": "487"}))
    provider, _ = _provider(sdk)

    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)

    assert sdk.calls == 1
    assert clock.sleeps == []
    assert excinfo.value.reason == "rate_limited"
    assert excinfo.value.retry_after == 487.0


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Please try again in 8m7.728s. Need more tokens?", 487.728),
        ("Rate limit reached. Please try again in 2.5s.", 2.5),
        ("Please try again in 559ms.", 0.559),
        ("Please try again in 1h2m3s.", 3723.0),
    ],
)
async def test_retry_after_is_read_from_groqs_message_when_the_header_is_missing(
    clock: FakeClock, message: str, expected: float
) -> None:
    sdk = _FakeSDK(_status_error(429, message=message))
    provider, _ = _provider(sdk, retry_max_wait_seconds=1000)
    # a 1s max_wait would fail fast; use a huge one and a big budget to just read it
    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)
    assert excinfo.value.retry_after == pytest.approx(expected, rel=1e-3)


async def test_server_errors_are_retried_to_the_limit(clock: FakeClock) -> None:
    sdk = _FakeSDK(_status_error(500))
    provider, _ = _provider(sdk)

    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)

    assert sdk.calls == 3
    assert excinfo.value.reason == "server_error"
    assert len(clock.sleeps) == 2


async def test_connection_failures_and_sdk_timeouts_are_retried(clock: FakeClock) -> None:
    connection = _FakeSDK(groq.APIConnectionError(request=_REQUEST))
    provider, _ = _provider(connection)
    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)
    assert (connection.calls, excinfo.value.reason) == (3, "connection_error")

    timeout = _FakeSDK(groq.APITimeoutError(request=_REQUEST))
    provider, _ = _provider(timeout)
    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)
    assert (timeout.calls, excinfo.value.reason) == (3, "timeout")


async def test_the_retry_budget_bounds_total_time(clock: FakeClock) -> None:
    sdk = _FakeSDK(_status_error(500))
    provider, _ = _provider(sdk, retry_budget_seconds=0.6)  # room for one 0.5s backoff, not two

    with pytest.raises(_ProviderError):
        await provider.complete(_HI)

    assert sdk.calls == 2


async def test_the_breaker_opens_after_repeated_upstream_failures_then_stops_calling_groq(
    clock: FakeClock,
) -> None:
    sdk = _FakeSDK(_status_error(500))
    provider, _ = _provider(sdk)
    for _ in range(5):
        with pytest.raises(_ProviderError):
            await provider.complete(_HI)
    calls_when_opened = sdk.calls
    before = metric_value("medagent_llm_calls_total", provider="groq", outcome="circuit_open")

    with pytest.raises(_ProviderError) as excinfo:
        await provider.complete(_HI)

    assert sdk.calls == calls_when_opened  # failed fast: Groq was not called
    assert excinfo.value.reason == "circuit_open"
    after = metric_value("medagent_llm_calls_total", provider="groq", outcome="circuit_open")
    assert after - before == 1


async def test_rate_limits_and_bad_requests_never_open_the_breaker(clock: FakeClock) -> None:
    sdk = _FakeSDK(_status_error(429, {"retry-after": "600"}), _status_error(400))
    provider, _ = _provider(sdk)

    for _ in range(12):
        with pytest.raises(_ProviderError) as excinfo:
            await provider.complete(_HI)
        assert excinfo.value.reason != "circuit_open"

    assert sdk.calls == 12  # every call still reached Groq


async def test_token_usage_and_latency_are_recorded(clock: FakeClock) -> None:
    sdk = _FakeSDK(_ok_response())
    provider, _ = _provider(sdk)
    prompt_before = metric_value("medagent_llm_tokens_total", provider="groq", kind="prompt")
    ok_before = metric_value("medagent_llm_calls_total", provider="groq", outcome="success")

    await provider.complete(_HI)

    assert (
        metric_value("medagent_llm_tokens_total", provider="groq", kind="prompt") - prompt_before
        == 10
    )
    assert (
        metric_value("medagent_llm_calls_total", provider="groq", outcome="success") - ok_before
        == 1
    )

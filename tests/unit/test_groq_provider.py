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

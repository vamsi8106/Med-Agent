import json
from typing import Any

import httpx
import pytest

from medagent.core.exceptions import ProviderError
from medagent.core.models import Message, ToolCall
from medagent.llm.openai_compatible_provider import OpenAICompatibleProvider
from tests.conftest import FakeClock

_OK_BODY = {
    "choices": [
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_guidelines",
                            "arguments": json.dumps({"query": "type 2 diabetes"}),
                        },
                    }
                ],
            }
        }
    ],
    "usage": {"prompt_tokens": 120, "completion_tokens": 15},
}


def _provider(handler: Any, **kwargs: Any) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://ollama:11434/v1/",
        model="qwen2.5:3b",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_request_shape_and_parsed_response() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_OK_BODY)

    provider = _provider(handler, api_key="local-key")
    messages = [
        Message(role="user", content="next steps?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_0", tool_name="search_guidelines", arguments={"q": 1})],
        ),
        Message(role="tool", content="no results", tool_call_id="call_0"),
    ]
    tools = [{"type": "function", "function": {"name": "search_guidelines"}}]

    response = await provider.complete(messages, tools)

    assert seen["url"] == "http://ollama:11434/v1/chat/completions"
    assert seen["auth"] == "Bearer local-key"
    assert seen["body"]["model"] == "qwen2.5:3b"
    assert seen["body"]["tools"] == tools
    assert seen["body"]["messages"][1]["tool_calls"][0]["function"]["name"] == "search_guidelines"
    assert seen["body"]["messages"][2] == {
        "role": "tool",
        "content": "no results",
        "tool_call_id": "call_0",
    }
    assert response.tool_calls == [
        ToolCall(id="call_1", tool_name="search_guidelines", arguments={"query": "type 2 diabetes"})
    ]
    assert response.usage == {"prompt_tokens": 120, "completion_tokens": 15}


async def test_missing_tool_call_ids_are_filled_in() -> None:
    body = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "t", "arguments": "{}"}}],
                }
            }
        ]
    }
    provider = _provider(lambda request: httpx.Response(200, json=body))

    response = await provider.complete([Message(role="user", content="hi")])

    assert response.tool_calls[0].id == "call_0"


@pytest.mark.parametrize(
    ("status", "headers", "reason", "retryable"),
    [
        (429, {"retry-after": "2"}, "rate_limited", True),
        (503, {}, "server_error", True),
        (400, {}, "client_error", False),
        (404, {}, "client_error", False),
    ],
)
async def test_http_errors_are_classified(
    clock: FakeClock, status: int, headers: dict[str, str], reason: str, retryable: bool
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, headers=headers, json={"error": "x"})

    provider = _provider(handler)

    with pytest.raises(ProviderError) as raised:
        await provider.complete([Message(role="user", content="hi")])

    assert raised.value.reason == reason
    assert raised.value.retryable is retryable
    assert calls == (3 if retryable else 1)  # 4xx is attempted once, never retried


async def test_timeouts_and_connection_failures_are_retryable(clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderError) as raised:
        await _provider(handler).complete([Message(role="user", content="hi")])
    assert raised.value.reason == "connection_error"

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError) as raised:
        await _provider(slow).complete([Message(role="user", content="hi")])
    assert raised.value.reason == "timeout"


async def test_breaker_opens_after_repeated_upstream_failures(clock: FakeClock) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    provider = _provider(handler)
    for _ in range(5):  # the breaker counts one failure per request (retries are inside it)
        with pytest.raises(ProviderError):
            await provider.complete([Message(role="user", content="hi")])
    calls_before = calls

    with pytest.raises(ProviderError) as raised:
        await provider.complete([Message(role="user", content="hi")])

    assert raised.value.reason == "circuit_open"
    assert calls == calls_before
    assert provider.circuit_state == "open"


async def test_unexpected_response_shape_is_a_provider_error() -> None:
    provider = _provider(lambda request: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError):
        await provider.complete([Message(role="user", content="hi")])

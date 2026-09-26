"""Groq LLM provider. No Groq imports allowed outside this file."""

import asyncio
import json
import os
from typing import Any

from groq import AsyncGroq
from langsmith import traceable

from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message, ToolCall
from medagent.infra.retry import retry


def _to_groq_message(message: Message) -> dict[str, Any]:
    """Maps our Message onto Groq's chat format, preserving the tool-call
    protocol: an assistant turn lists the calls it made, and each tool result
    is a role="tool" message referencing its call id."""
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id or "",
        }
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["content"] = message.content or None
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.tool_name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    return payload


def _parse_tool_calls(raw_calls: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in raw_calls or []:
        try:
            arguments = json.loads(raw.function.arguments or "{}")
        except json.JSONDecodeError:
            # Malformed arguments become an empty call, which the tool rejects
            # with a clear error the model can see and correct next turn.
            arguments = {}
        calls.append(
            ToolCall(
                id=raw.id,
                tool_name=raw.function.name,
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


def _enable_langsmith_tracing(api_key: str | None, project: str) -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project


class GroqProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        tracing_enabled: bool = False,
        langsmith_api_key: str | None = None,
        langsmith_project: str = "medagent",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        self._timeout_seconds = timeout_seconds
        if tracing_enabled:
            _enable_langsmith_tracing(langsmith_api_key, langsmith_project)

    # langsmith's @traceable wraps this in a protocol type that mypy sees as
    # an incompatible override of BaseLLMProvider.complete, even though it's
    # behaviorally the same coroutine at runtime (exercised by the passing
    # unit and eval suites).
    @traceable(run_type="llm", name="groq_chat_completion")
    @retry(max_attempts=3, exceptions=(Exception,))
    async def complete(  # type: ignore[override]
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    # Groq's SDK wants a union of specific per-role TypedDicts;
                    # our provider-agnostic Message model doesn't map 1:1, but
                    # the plain dict shape built here is exactly what the API
                    # expects at runtime.
                    messages=[_to_groq_message(m) for m in messages],  # type: ignore[misc]
                    tools=tools,  # type: ignore[arg-type]
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise ProviderError(
                f"Groq completion timed out after {self._timeout_seconds}s"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"Groq completion failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=self._model,
            tool_calls=_parse_tool_calls(getattr(choice.message, "tool_calls", None)),
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        )

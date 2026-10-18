"""Groq LLM provider. No Groq imports allowed outside this file.

Reliability lives here, in one place. The SDK's own retries are switched off
(max_retries=0): left on, they stacked under ours -- up to 9 HTTP calls per LLM
call -- and wrapped nothing we could observe or classify. Errors are mapped to
ProviderError with a reason, a retryable flag and the wait Groq asked for, so
the retry policy and the circuit breaker can act on them correctly: a 429 is
retried (or failed fast if it needs a long wait), a 404 or 400 is never
retried, and only genuine upstream trouble (timeouts, connection failures, 5xx)
counts toward opening the breaker.
"""

import asyncio
import os
import re
from typing import Any

import groq
from groq import AsyncGroq

from medagent.core.exceptions import ProviderError
from medagent.core.models import Message
from medagent.llm.chat_base import (
    ChatCompletionProvider,
    ChatResult,
    classify_status,
    parse_tool_calls,
    retry_after_from_headers,
    to_chat_message,
)

_RETRY_HINT = re.compile(r"try again in ([0-9hms.\s]+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _seconds_from_message(message: str) -> float | None:
    """Groq's 429 text says how long to wait ("try again in 2.5s", "8m7.728s",
    "559ms") -- used when the Retry-After header is absent."""
    hint = _RETRY_HINT.search(message)
    if hint is None:
        return None
    parts = _DURATION_PART.findall(hint.group(1))
    if not parts:
        return None
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in parts)


def _retry_after(exc: groq.APIStatusError) -> float | None:
    from_headers = retry_after_from_headers(exc.response.headers)
    return from_headers if from_headers is not None else _seconds_from_message(str(exc))


def _classify_status_error(exc: groq.APIStatusError) -> ProviderError:
    return classify_status(exc.status_code, f"Groq completion failed: {exc}", _retry_after(exc))


def _tool_calls_as_dicts(raw_calls: Any) -> list[dict[str, Any]]:
    return [
        {
            "id": raw.id,
            "function": {"name": raw.function.name, "arguments": raw.function.arguments},
        }
        for raw in raw_calls or []
    ]


def _enable_langsmith_tracing(api_key: str | None, project: str) -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project


class GroqProvider(ChatCompletionProvider):
    provider_name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str,
        tracing_enabled: bool = False,
        langsmith_api_key: str | None = None,
        langsmith_project: str = "medagent",
        timeout_seconds: float = 30.0,
        retry_max_wait_seconds: float = 10.0,
        retry_budget_seconds: float = 45.0,
    ) -> None:
        super().__init__(
            model,
            timeout_seconds=timeout_seconds,
            retry_max_wait_seconds=retry_max_wait_seconds,
            retry_budget_seconds=retry_budget_seconds,
        )
        self._client = AsyncGroq(api_key=api_key, max_retries=0)
        if tracing_enabled:
            _enable_langsmith_tracing(langsmith_api_key, langsmith_project)

    async def _call_once(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> ChatResult:
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    # Groq's SDK wants a union of specific per-role TypedDicts;
                    # the plain OpenAI-format dicts built here are exactly what
                    # the API expects at runtime.
                    messages=[to_chat_message(m) for m in messages],  # type: ignore[misc]
                    tools=tools,  # type: ignore[arg-type]
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise ProviderError(
                f"Groq completion timed out after {self._timeout_seconds}s",
                reason="timeout",
                retryable=True,
            ) from exc
        except groq.APIStatusError as exc:
            raise _classify_status_error(exc) from exc
        except groq.APITimeoutError as exc:
            raise ProviderError(
                f"Groq completion timed out: {exc}", reason="timeout", retryable=True
            ) from exc
        except groq.APIConnectionError as exc:
            raise ProviderError(
                f"Groq connection failed: {exc}", reason="connection_error", retryable=True
            ) from exc
        except Exception as exc:
            raise ProviderError(f"Groq completion failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return ChatResult(
            content=choice.message.content or "",
            tool_calls=parse_tool_calls(
                _tool_calls_as_dicts(getattr(choice.message, "tool_calls", None))
            ),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

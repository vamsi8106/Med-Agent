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
import json
import os
import re
import time
from typing import Any

import groq
from groq import AsyncGroq
from langsmith import traceable

from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message, ToolCall
from medagent.infra.circuit_breaker import CircuitBreaker
from medagent.infra.metrics import llm_call_duration_seconds, llm_calls_total, llm_tokens_total
from medagent.infra.retry import retry
from medagent.infra.tracing import get_tracer

_PROVIDER = "groq"
tracer = get_tracer(__name__)
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
    headers = exc.response.headers
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(header)
        if raw is not None:
            try:
                return float(raw) * scale
            except ValueError:
                continue
    return _seconds_from_message(str(exc))


def _classify_status_error(exc: groq.APIStatusError) -> ProviderError:
    message = f"Groq completion failed: {exc}"
    status = exc.status_code
    if status == 429:
        return ProviderError(
            message, reason="rate_limited", retryable=True, retry_after=_retry_after(exc)
        )
    if status in (408, 409) or status >= 500:
        return ProviderError(message, reason="server_error", retryable=True)
    # 400/401/403/404/413/422...: the request itself is wrong. Retrying cannot help.
    return ProviderError(message, reason="client_error")


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
        retry_max_wait_seconds: float = 10.0,
        retry_budget_seconds: float = 45.0,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key, max_retries=0)
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._breaker = CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0, name=_PROVIDER, error_type=ProviderError
        )
        self._call_with_retry = retry(
            max_attempts=3,
            max_wait=retry_max_wait_seconds,
            budget=retry_budget_seconds,
            target="llm",
        )(self._call_once)
        if tracing_enabled:
            _enable_langsmith_tracing(langsmith_api_key, langsmith_project)

    @traceable(run_type="llm", name="groq_chat_completion")
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        started = time.monotonic()
        # OTel GenAI attributes: model, token counts and outcome -- never the
        # prompt or the completion, which carry patient data.
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("gen_ai.system", _PROVIDER)
            span.set_attribute("gen_ai.request.model", self._model)
            try:
                response = await self._breaker.call(self._call_with_retry, messages, tools)
            except ProviderError as exc:
                span.set_attribute("medagent.llm.outcome", exc.reason)
                self._record_call(exc.reason, started)
                raise
            self._record_call("success", started)

            choice = response.choices[0]
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            span.set_attribute("medagent.llm.outcome", "success")
            span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
        llm_tokens_total.labels(provider=_PROVIDER, kind="prompt").inc(prompt_tokens)
        llm_tokens_total.labels(provider=_PROVIDER, kind="completion").inc(completion_tokens)
        return LLMResponse(
            content=choice.message.content or "",
            model=self._model,
            tool_calls=_parse_tool_calls(getattr(choice.message, "tool_calls", None)),
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        )

    @property
    def circuit_state(self) -> str:
        return self._breaker.state.value

    def _record_call(self, outcome: str, started: float) -> None:
        llm_calls_total.labels(provider=_PROVIDER, outcome=outcome).inc()
        llm_call_duration_seconds.labels(provider=_PROVIDER).observe(time.monotonic() - started)

    async def _call_once(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> Any:
        try:
            return await asyncio.wait_for(
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

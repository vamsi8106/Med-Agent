"""What every chat-completion provider shares, whoever hosts the model.

Reliability and observability live here once: the circuit breaker, the retry
policy, the `llm.chat` span (model, token counts, outcome -- never the prompt
or the completion, which carry patient data) and the LLM metrics. A provider
subclass only knows how to make one call and classify its errors; it returns a
ChatResult and raises ProviderError with a reason and a retryable flag.

The message helpers speak the OpenAI chat format, which both Groq and the
self-hosted servers (Ollama, vLLM, LM Studio) accept.
"""

import json
import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langsmith import traceable

from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message, ToolCall
from medagent.infra.circuit_breaker import CircuitBreaker
from medagent.infra.metrics import llm_call_duration_seconds, llm_calls_total, llm_tokens_total
from medagent.infra.retry import retry
from medagent.infra.tracing import get_tracer

tracer = get_tracer(__name__)


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def to_chat_message(message: Message) -> dict[str, Any]:
    """Our Message in the OpenAI chat format, preserving the tool-call protocol:
    an assistant turn lists the calls it made, and each tool result is a
    role="tool" message referencing its call id."""
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


def parse_tool_calls(raw_calls: Sequence[Mapping[str, Any]] | None) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls or []):
        function = raw.get("function") or {}
        arguments_raw = function.get("arguments") or "{}"
        try:
            arguments = (
                arguments_raw if isinstance(arguments_raw, dict) else json.loads(arguments_raw)
            )
        except json.JSONDecodeError:
            # Malformed arguments become an empty call, which the tool rejects
            # with a clear error the model can see and correct next turn.
            arguments = {}
        calls.append(
            ToolCall(
                # Some local servers omit the id; the tool-result message must
                # still reference one, so make it unique within the turn.
                id=raw.get("id") or f"call_{index}",
                tool_name=function.get("name") or "",
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return calls


def retry_after_from_headers(headers: Mapping[str, str]) -> float | None:
    for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(header)
        if raw is not None:
            try:
                return float(raw) * scale
            except ValueError:
                continue
    return None


def classify_status(status: int, message: str, retry_after: float | None) -> ProviderError:
    if status == 429:
        return ProviderError(
            message, reason="rate_limited", retryable=True, retry_after=retry_after
        )
    if status in (408, 409) or status >= 500:
        return ProviderError(message, reason="server_error", retryable=True)
    # 400/401/403/404/413/422...: the request itself is wrong. Retrying cannot help.
    return ProviderError(message, reason="client_error")


class ChatCompletionProvider(BaseLLMProvider):
    provider_name: str  # metric/span label; one of a small fixed set

    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: float,
        retry_max_wait_seconds: float,
        retry_budget_seconds: float,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            name=self.provider_name,
            error_type=ProviderError,
        )
        self._call_with_retry = retry(
            max_attempts=3,
            max_wait=retry_max_wait_seconds,
            budget=retry_budget_seconds,
            target="llm",
        )(self._call_once)

    @abstractmethod
    async def _call_once(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> ChatResult: ...

    @traceable(run_type="llm", name="chat_completion")
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        started = time.monotonic()
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("gen_ai.system", self.provider_name)
            span.set_attribute("gen_ai.request.model", self._model)
            try:
                result = await self._breaker.call(self._call_with_retry, messages, tools)
            except ProviderError as exc:
                span.set_attribute("medagent.llm.outcome", exc.reason)
                self._record_call(exc.reason, started)
                raise
            self._record_call("success", started)
            span.set_attribute("medagent.llm.outcome", "success")
            span.set_attribute("gen_ai.usage.input_tokens", result.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", result.completion_tokens)
        llm_tokens_total.labels(provider=self.provider_name, kind="prompt").inc(
            result.prompt_tokens
        )
        llm_tokens_total.labels(provider=self.provider_name, kind="completion").inc(
            result.completion_tokens
        )
        return LLMResponse(
            content=result.content,
            model=self._model,
            tool_calls=result.tool_calls,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        )

    @property
    def circuit_state(self) -> str:
        return self._breaker.state.value

    def _record_call(self, outcome: str, started: float) -> None:
        llm_calls_total.labels(provider=self.provider_name, outcome=outcome).inc()
        llm_call_duration_seconds.labels(provider=self.provider_name).observe(
            time.monotonic() - started
        )

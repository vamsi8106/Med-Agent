"""Provider for any server that speaks the OpenAI chat-completions API.

Meant for a model you host yourself -- Ollama, vLLM, LM Studio -- so prompts,
which carry patient data, never leave your infrastructure. Plain httpx, no
vendor SDK. Errors map onto the same ProviderError reasons as Groq, so the
retry policy and circuit breaker in chat_base behave identically.
"""

from typing import Any

import httpx

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


class OpenAICompatibleProvider(ChatCompletionProvider):
    provider_name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        retry_max_wait_seconds: float = 10.0,
        retry_budget_seconds: float = 45.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(
            model,
            timeout_seconds=timeout_seconds,
            retry_max_wait_seconds=retry_max_wait_seconds,
            retry_budget_seconds=retry_budget_seconds,
        )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_seconds,
            transport=transport,
        )

    async def _call_once(
        self, messages: list[Message], tools: list[dict[str, Any]] | None
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [to_chat_message(m) for m in messages],
        }
        if tools:
            body["tools"] = tools
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"LLM completion timed out after {self._timeout_seconds}s",
                reason="timeout",
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"LLM connection failed: {exc}", reason="connection_error", retryable=True
            ) from exc

        if response.status_code >= 400:
            raise classify_status(
                response.status_code,
                f"LLM completion failed with HTTP {response.status_code}",
                retry_after_from_headers(response.headers),
            )
        try:
            payload = response.json()
            message = payload["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("LLM returned an unexpected response shape") from exc

        usage = payload.get("usage") or {}
        return ChatResult(
            content=message.get("content") or "",
            tool_calls=parse_tool_calls(message.get("tool_calls")),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )

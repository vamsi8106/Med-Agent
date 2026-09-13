"""Groq LLM provider. No Groq imports allowed outside this file."""

from typing import Any

from groq import AsyncGroq

from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message
from medagent.infra.retry import retry


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @retry(max_attempts=3, exceptions=(Exception,))
    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                tools=tools,
            )
        except Exception as exc:
            raise ProviderError(f"Groq completion failed: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=self._model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
            },
        )

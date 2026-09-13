"""Groq LLM provider. No Groq imports allowed outside this file."""

import os
from typing import Any

from groq import AsyncGroq
from langsmith import traceable

from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message
from medagent.infra.retry import retry


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
    ) -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model
        if tracing_enabled:
            _enable_langsmith_tracing(langsmith_api_key, langsmith_project)

    @traceable(run_type="llm", name="groq_chat_completion")
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

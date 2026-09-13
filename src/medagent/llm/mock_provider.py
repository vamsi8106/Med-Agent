"""Deterministic LLM provider for tests. No network calls."""

from typing import Any

from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, fixed_response: str = "mock response") -> None:
        self._fixed_response = fixed_response

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        return LLMResponse(content=self._fixed_response, model="mock-model")

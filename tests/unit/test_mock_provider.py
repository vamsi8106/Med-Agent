import pytest

from medagent.core.models import Message
from medagent.llm.mock_provider import MockLLMProvider


@pytest.mark.asyncio
async def test_mock_provider_returns_fixed_response() -> None:
    provider = MockLLMProvider(fixed_response="hello")
    response = await provider.complete([Message(role="user", content="hi")])
    assert response.content == "hello"
    assert response.model == "mock-model"

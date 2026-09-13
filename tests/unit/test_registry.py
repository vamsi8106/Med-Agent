import pytest

from medagent.core.config import Settings
from medagent.core.exceptions import ProviderError
from medagent.llm.mock_provider import MockLLMProvider
from medagent.llm.registry import ProviderRegistry


def test_registry_returns_mock_provider() -> None:
    provider = ProviderRegistry.get_provider("mock")
    assert isinstance(provider, MockLLMProvider)


def test_registry_unknown_provider_raises() -> None:
    with pytest.raises(ProviderError):
        ProviderRegistry.get_provider("unknown")


def test_registry_groq_without_api_key_raises() -> None:
    settings = Settings(_env_file=None, groq_api_key=None)
    with pytest.raises(ProviderError):
        ProviderRegistry.get_provider("groq", settings=settings)

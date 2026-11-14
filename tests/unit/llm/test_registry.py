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


def test_registry_openai_compatible_requires_base_url_and_model() -> None:
    with pytest.raises(ProviderError, match="LLM_BASE_URL"):
        ProviderRegistry.get_provider("openai_compatible", Settings(_env_file=None))


def test_registry_returns_openai_compatible_provider() -> None:
    from medagent.llm.openai_compatible_provider import OpenAICompatibleProvider

    provider = ProviderRegistry.get_provider(
        "openai_compatible",
        Settings(_env_file=None, llm_base_url="http://localhost:11434/v1", llm_model="qwen2.5:3b"),
    )
    assert isinstance(provider, OpenAICompatibleProvider)

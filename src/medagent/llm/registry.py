"""Provider registry: resolves LLM_PROVIDER setting to a BaseLLMProvider instance."""

from medagent.core.config import Settings
from medagent.core.exceptions import ProviderError
from medagent.core.interfaces import BaseLLMProvider
from medagent.llm.mock_provider import MockLLMProvider


class ProviderRegistry:
    @staticmethod
    def get_provider(name: str, settings: Settings | None = None) -> BaseLLMProvider:
        settings = settings or Settings()

        if name == "mock":
            return MockLLMProvider()

        if name == "groq":
            from medagent.llm.groq_provider import GroqProvider

            if not settings.groq_api_key:
                raise ProviderError("GROQ_API_KEY is not configured")
            return GroqProvider(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                tracing_enabled=settings.langchain_tracing_v2,
                langsmith_api_key=settings.langchain_api_key,
                langsmith_project=settings.langchain_project,
                timeout_seconds=settings.llm_timeout_seconds,
            )

        raise ProviderError(f"Unknown LLM provider: {name}")

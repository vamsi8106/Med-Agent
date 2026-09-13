import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from medagent.core.exceptions import ProviderError
from medagent.core.models import Message
from medagent.llm.groq_provider import GroqProvider


def test_tracing_disabled_by_default_leaves_env_untouched() -> None:
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    with patch("medagent.llm.groq_provider.AsyncGroq", return_value=MagicMock()):
        GroqProvider(api_key="test-key", model="test-model")
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_tracing_enabled_sets_langsmith_env_vars() -> None:
    try:
        with patch("medagent.llm.groq_provider.AsyncGroq", return_value=MagicMock()):
            GroqProvider(
                api_key="test-key",
                model="test-model",
                tracing_enabled=True,
                langsmith_api_key="ls-key",
                langsmith_project="medagent-test",
            )
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-key"
        assert os.environ["LANGCHAIN_PROJECT"] == "medagent-test"
    finally:
        for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
            os.environ.pop(key, None)


async def test_complete_raises_provider_error_on_timeout() -> None:
    async def _hangs_forever(**_kwargs: object) -> None:
        await asyncio.sleep(10)

    fake_client = MagicMock()
    fake_client.chat.completions.create = _hangs_forever

    with patch("medagent.llm.groq_provider.AsyncGroq", return_value=fake_client):
        provider = GroqProvider(api_key="test-key", model="test-model", timeout_seconds=0.05)

    with pytest.raises(ProviderError, match="timed out"):
        await provider.complete([Message(role="user", content="hi")])

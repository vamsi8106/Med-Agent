"""With LANGFUSE_ENABLED=true (and keys) in .env, every eval run's LLM calls --
messages, answers, tool calls, tokens -- go to Langfuse, so a failing case can
be read there. The evals build agents directly, without create_app, so this
attaches Langfuse to the test session's tracer provider instead."""

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from medagent.core.config import get_settings
from medagent.infra.tracing import attach_langfuse, shutdown_tracing


@pytest.fixture(scope="session", autouse=True)
def langfuse_for_evals() -> Iterator[None]:
    settings = get_settings()
    provider = trace.get_tracer_provider()
    if not settings.langfuse_enabled or not isinstance(provider, TracerProvider):
        yield
        return
    attach_langfuse(provider, settings)
    yield
    shutdown_tracing()  # flushes Langfuse; the session's own provider is left alone

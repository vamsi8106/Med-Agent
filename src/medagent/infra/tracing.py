"""OpenTelemetry tracing setup and helpers.

Spans are always created: they are cheap, and they carry the trace_id that ties
log lines to a request. They are exported only when OTEL_EXPORTER_OTLP_ENDPOINT
is set (OTLP/HTTP, e.g. to Tempo); OTEL_CONSOLE_EXPORTER prints them to stdout
for local debugging and is off by default.

Telemetry rule: spans (names, attributes, events) never carry patient data --
no patient id, name, medication, or prompt/response text. Use route templates,
counts, model names, token counts and outcomes.

One exception, opt-in: with LANGFUSE_ENABLED, `llm_generation` records each LLM
call's messages and answer on a span of Langfuse's own scope (`langfuse-sdk`).
Langfuse is attached to the same provider (one tree: route -> nodes -> LLM
calls), but the OTLP and console exports drop that scope (`_ExcludeScopes`), so
prompt text reaches Langfuse only -- never Tempo, Grafana or the logs.
"""

import importlib.metadata
import threading
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from medagent.core.config import Settings, get_settings

# Langfuse's tracer name: the only scope whose spans may carry prompt text.
LANGFUSE_SCOPE = "langfuse-sdk"

_lock = threading.Lock()
_owned_provider: TracerProvider | None = None
_langfuse: Langfuse | None = None


class _ExcludeScopes(SpanProcessor):
    """Passes spans to `inner` unless they come from one of `scopes`. Wraps the
    OTLP and console exports so Langfuse generations (prompt text) never reach
    them."""

    def __init__(self, inner: SpanProcessor, scopes: frozenset[str]) -> None:
        self.inner = inner
        self._scopes = scopes

    def _excluded(self, span: ReadableSpan) -> bool:
        scope = span.instrumentation_scope
        return scope is not None and scope.name in self._scopes

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        if not self._excluded(span):
            self.inner.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        if not self._excluded(span):
            self.inner.on_end(span)

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.inner.force_flush(timeout_millis)


def _without_prompts(processor: SpanProcessor) -> SpanProcessor:
    return _ExcludeScopes(processor, frozenset({LANGFUSE_SCOPE}))


def _service_version() -> str:
    try:
        return importlib.metadata.version("medagent")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def configure_tracing(
    settings: Settings | None = None, service_name: str = "medagent"
) -> TracerProvider | None:
    """Installs the global tracer provider. Returns it, or None if one was
    already installed (an earlier app start, or a test harness) -- OpenTelemetry
    only allows setting the provider once per process, and we never override
    one we did not create."""
    global _owned_provider
    settings = settings or get_settings()
    with _lock:
        if isinstance(trace.get_tracer_provider(), TracerProvider):
            return None

        provider = TracerProvider(
            resource=Resource.create(
                {SERVICE_NAME: service_name, SERVICE_VERSION: _service_version()}
            ),
            sampler=ParentBased(TraceIdRatioBased(settings.otel_sample_ratio)),
        )
        endpoint = settings.otel_exporter_otlp_endpoint
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
            provider.add_span_processor(_without_prompts(BatchSpanProcessor(exporter)))
        if settings.otel_console_exporter:
            provider.add_span_processor(_without_prompts(BatchSpanProcessor(ConsoleSpanExporter())))

        trace.set_tracer_provider(provider)
        _owned_provider = provider
    if settings.langfuse_enabled:
        attach_langfuse(provider, settings)
    return provider


def langfuse_attached() -> bool:
    """Whether prompt tracing to Langfuse is on in this process."""
    return _langfuse is not None


def attach_langfuse(
    provider: TracerProvider,
    settings: Settings | None = None,
    *,
    span_exporter: SpanExporter | None = None,
) -> Langfuse:
    """Sends every span of `provider` to Langfuse, including the generations
    `llm_generation` records. Exports already on `provider` that were not added
    by configure_tracing (a test harness's) are not filtered. `span_exporter`
    replaces the network export (tests)."""
    global _langfuse
    settings = settings or get_settings()
    with _lock:
        if _langfuse is None:
            _langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
                environment=settings.app_env,
                release=_service_version(),
                tracer_provider=provider,
                # Ours carry no patient data, and with them the tree is whole:
                # route -> graph nodes -> ReAct turns -> llm.chat -> generation.
                should_export_span=lambda span: True,
                span_exporter=span_exporter,
            )
        return _langfuse


def shutdown_tracing() -> None:
    """Flushes and stops the provider this process created. A provider someone
    else installed is left alone."""
    global _owned_provider, _langfuse
    with _lock:
        if _langfuse is not None:
            _langfuse.shutdown()
            _langfuse = None
        if _owned_provider is not None:
            _owned_provider.shutdown()
            _owned_provider = None


class Generation:
    """One LLM call as Langfuse sees it. Does nothing when Langfuse is off."""

    def __init__(self, observation: Any | None) -> None:
        self._observation = observation

    def finish(self, output: dict[str, Any], usage: dict[str, int]) -> None:
        if self._observation is not None:
            self._observation.update(output=output, usage_details=usage)

    def fail(self, reason: str) -> None:
        if self._observation is not None:
            self._observation.update(level="ERROR", status_message=reason)


@contextmanager
def llm_generation(
    name: str, *, model: str, messages: list[dict[str, Any]], metadata: dict[str, Any]
) -> Iterator[Generation]:
    """Records one LLM call -- its messages and, via the handle, its answer --
    as a Langfuse generation. These are the only spans allowed to carry prompt
    text, and they reach Langfuse only (module docstring)."""
    client = _langfuse
    if client is None:
        yield Generation(None)
        return
    with client.start_as_current_observation(
        name=name, as_type="generation", model=model, input=messages, metadata=metadata
    ) as observation:
        yield Generation(observation)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def traced_node(
    name: str, node: Callable[..., Coroutine[Any, Any, dict[str, Any]]]
) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    """Wraps a LangGraph node in a `node.<name>` span, so one assessment shows up
    as a tree (triage, each specialist, the report) instead of an opaque call."""
    tracer = get_tracer("medagent.workflow")

    async def wrapper(state: Any) -> dict[str, Any]:
        with tracer.start_as_current_span(f"node.{name}"):
            return await node(state)

    return wrapper

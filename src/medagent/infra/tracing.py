"""OpenTelemetry tracing setup and helpers.

Spans are always created: they are cheap, and they carry the trace_id that ties
log lines to a request. They are exported only when OTEL_EXPORTER_OTLP_ENDPOINT
is set (OTLP/HTTP, e.g. to Tempo); OTEL_CONSOLE_EXPORTER prints them to stdout
for local debugging and is off by default.

Telemetry rule: spans (names, attributes, events) never carry patient data --
no patient id, name, medication, or prompt/response text. Use route templates,
counts, model names, token counts and outcomes.
"""

import importlib.metadata
import threading
from collections.abc import Callable, Coroutine
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from medagent.core.config import Settings, get_settings

_lock = threading.Lock()
_owned_provider: TracerProvider | None = None


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
            provider.add_span_processor(BatchSpanProcessor(exporter))
        if settings.otel_console_exporter:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _owned_provider = provider
        return provider


def shutdown_tracing() -> None:
    """Flushes and stops the provider this process created. A provider someone
    else installed is left alone."""
    global _owned_provider
    with _lock:
        if _owned_provider is not None:
            _owned_provider.shutdown()
            _owned_provider = None


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

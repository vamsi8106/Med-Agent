from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from medagent.core.config import Settings
from medagent.infra import tracing
from medagent.infra.logging import add_trace_context
from medagent.infra.tracing import configure_tracing, shutdown_tracing, traced_node


@pytest.fixture
def fresh_process(monkeypatch: pytest.MonkeyPatch) -> list[TracerProvider]:
    """Makes configure_tracing() believe no provider is installed yet, and
    captures what it installs (the real global can only be set once)."""
    installed: list[TracerProvider] = []
    monkeypatch.setattr(tracing.trace, "get_tracer_provider", lambda: trace.ProxyTracerProvider())
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", installed.append)
    monkeypatch.setattr(tracing, "_owned_provider", None)
    return installed


def _exporters(provider: TracerProvider) -> list[type]:
    # private attribute: the SDK offers no public way to list processors
    processors = provider._active_span_processor._span_processors
    return [type(p.span_exporter) for p in processors]


def test_no_exporter_is_attached_unless_an_endpoint_is_configured(
    fresh_process: list[TracerProvider],
) -> None:
    provider = configure_tracing(Settings(_env_file=None))

    assert provider is fresh_process[0]
    assert _exporters(provider) == []  # spans still exist (for trace ids), just aren't shipped


def test_an_endpoint_enables_otlp_http_export_to_the_traces_path(
    fresh_process: list[TracerProvider],
) -> None:
    provider = configure_tracing(
        Settings(_env_file=None, otel_exporter_otlp_endpoint="http://tempo:4318/")
    )

    assert _exporters(provider) == [OTLPSpanExporter]
    exporter = provider._active_span_processor._span_processors[0].span_exporter
    assert exporter._endpoint == "http://tempo:4318/v1/traces"


def test_the_console_exporter_is_opt_in(fresh_process: list[TracerProvider]) -> None:
    provider = configure_tracing(Settings(_env_file=None, otel_console_exporter=True))
    assert _exporters(provider) == [ConsoleSpanExporter]


def test_the_sample_ratio_is_applied(fresh_process: list[TracerProvider]) -> None:
    provider = configure_tracing(Settings(_env_file=None, otel_sample_ratio=0.25))
    assert "0.25" in provider.sampler.get_description()


def test_the_resource_names_the_service(fresh_process: list[TracerProvider]) -> None:
    provider = configure_tracing(Settings(_env_file=None))
    assert provider.resource.attributes["service.name"] == "medagent"
    assert provider.resource.attributes["service.version"]


def test_an_already_installed_provider_is_never_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenTelemetry allows setting the provider once per process; repeated app
    starts (and the test harness's own provider) must be left alone."""
    monkeypatch.setattr(tracing.trace, "get_tracer_provider", lambda: TracerProvider())
    set_provider = MagicMock()
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", set_provider)

    assert configure_tracing(Settings(_env_file=None)) is None
    set_provider.assert_not_called()


def test_shutdown_flushes_only_a_provider_we_created(monkeypatch: pytest.MonkeyPatch) -> None:
    ours = MagicMock()
    monkeypatch.setattr(tracing, "_owned_provider", ours)

    shutdown_tracing()
    ours.shutdown.assert_called_once()
    assert tracing._owned_provider is None

    shutdown_tracing()  # nothing owned now: a provider someone else installed is untouched
    ours.shutdown.assert_called_once()


# --- log correlation ------------------------------------------------------------------


def test_log_lines_inside_a_span_carry_its_trace_and_span_ids(spans) -> None:
    with trace.get_tracer("t").start_as_current_span("work") as span:
        event = add_trace_context(None, "info", {"event": "something_happened"})
        context = span.get_span_context()

    assert event["trace_id"] == format(context.trace_id, "032x")
    assert event["span_id"] == format(context.span_id, "016x")


def test_log_lines_outside_any_span_have_no_trace_ids() -> None:
    event = add_trace_context(None, "info", {"event": "startup"})
    assert "trace_id" not in event
    assert "span_id" not in event


async def test_traced_node_wraps_a_node_in_a_named_span(spans) -> None:
    async def node(state: dict) -> dict:
        return {"answer": 1}

    assert await traced_node("triage", node)({}) == {"answer": 1}

    assert [s.name for s in spans.get_finished_spans()] == ["node.triage"]


async def test_traced_node_records_an_exception_and_reraises(spans) -> None:
    async def node(state: dict) -> dict:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await traced_node("evidence", node)({})

    span = spans.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert any(event.name == "exception" for event in span.events)

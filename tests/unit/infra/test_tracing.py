import uuid
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medagent.core.config import Settings
from medagent.infra import tracing
from medagent.infra.logging import add_trace_context
from medagent.infra.tracing import (
    LANGFUSE_SCOPE,
    attach_langfuse,
    configure_tracing,
    llm_generation,
    shutdown_tracing,
    traced_node,
)
from tests.conftest import span_text


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
    # Our exports sit inside _ExcludeScopes, which keeps Langfuse's prompt spans out.
    return [type(getattr(p, "inner", p).span_exporter) for p in processors]


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
    exporter = provider._active_span_processor._span_processors[0].inner.span_exporter
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
    monkeypatch.setattr(tracing, "_langfuse", None)

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


# --- Langfuse: prompts reach Langfuse only ---------------------------------------------


def _langfuse_settings(**overrides: object) -> Settings:
    # A fresh public key per test: the SDK keeps one client per key.
    return Settings(
        _env_file=None,
        langfuse_enabled=True,
        langfuse_public_key=f"pk-lf-test-{uuid.uuid4().hex}",
        langfuse_secret_key="sk-lf-test",
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
def no_langfuse(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(tracing, "_langfuse", None)
    yield
    if tracing._langfuse is not None:
        tracing._langfuse.shutdown()
    tracing._langfuse = None


def test_prompt_text_reaches_langfuse_but_never_the_otlp_export(no_langfuse: None) -> None:
    provider = TracerProvider()
    tempo, langfuse = InMemorySpanExporter(), InMemorySpanExporter()
    provider.add_span_processor(tracing._without_prompts(SimpleSpanProcessor(tempo)))
    client = attach_langfuse(provider, _langfuse_settings(), span_exporter=langfuse)
    secret = "Patient Alpha zzyzx-sentinel"

    with provider.get_tracer("medagent.workflow").start_as_current_span("node.evidence"):
        with llm_generation(
            "chat_completion",
            model="test-model",
            messages=[{"role": "user", "content": secret}],
            metadata={"provider": "groq", "tools": []},
        ) as generation:
            generation.finish({"content": f"answer about {secret}"}, {"input": 3, "output": 4})
    client.flush()

    to_tempo = tempo.get_finished_spans()
    to_langfuse = langfuse.get_finished_spans()
    assert [s.name for s in to_tempo] == ["node.evidence"]
    assert not any("zzyzx" in span_text(s) for s in to_tempo)
    generations = [s for s in to_langfuse if s.instrumentation_scope.name == LANGFUSE_SCOPE]
    assert len(generations) == 1 and "zzyzx" in span_text(generations[0])
    # The whole tree goes to Langfuse: our span is the generation's parent.
    node = next(s for s in to_langfuse if s.name == "node.evidence")
    assert generations[0].parent.span_id == node.context.span_id


def test_langfuse_attached_reports_whether_prompt_tracing_is_on(no_langfuse: None) -> None:
    assert not tracing.langfuse_attached()
    attach_langfuse(TracerProvider(), _langfuse_settings(), span_exporter=InMemorySpanExporter())
    assert tracing.langfuse_attached()


def test_llm_generation_is_a_no_op_when_langfuse_is_off(spans, no_langfuse: None) -> None:
    with llm_generation("chat_completion", model="m", messages=[], metadata={}) as generation:
        generation.finish({"content": "x"}, {"input": 1, "output": 1})
        generation.fail("client_error")

    assert spans.get_finished_spans() == ()


def test_configure_tracing_attaches_langfuse_to_its_provider_when_enabled(
    fresh_process: list[TracerProvider], monkeypatch: pytest.MonkeyPatch, no_langfuse: None
) -> None:
    client_class = MagicMock()
    monkeypatch.setattr(tracing, "Langfuse", client_class)
    settings = _langfuse_settings(langfuse_base_url="https://us.cloud.langfuse.com")

    provider = configure_tracing(settings)

    kwargs = client_class.call_args.kwargs
    assert kwargs["tracer_provider"] is provider
    assert kwargs["base_url"] == "https://us.cloud.langfuse.com"
    assert kwargs["public_key"] == settings.langfuse_public_key
    tracing._langfuse = None  # a mock: nothing to shut down


def test_configure_tracing_leaves_langfuse_off_by_default(
    fresh_process: list[TracerProvider], monkeypatch: pytest.MonkeyPatch, no_langfuse: None
) -> None:
    client_class = MagicMock()
    monkeypatch.setattr(tracing, "Langfuse", client_class)

    configure_tracing(Settings(_env_file=None))

    client_class.assert_not_called()


def test_shutdown_flushes_the_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    monkeypatch.setattr(tracing, "_langfuse", client)
    monkeypatch.setattr(tracing, "_owned_provider", None)

    shutdown_tracing()

    client.shutdown.assert_called_once()
    assert tracing._langfuse is None

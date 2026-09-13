from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medagent.infra.middleware import ObservabilityMiddleware


def _make_app_with_span_capture() -> tuple[TestClient, InMemorySpanExporter, object]:
    # A private TracerProvider, not the global one -- OpenTelemetry only
    # allows the global provider to be set once per process, and test order
    # shouldn't determine whether this test can observe spans.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    app = FastAPI()
    app.add_middleware(ObservabilityMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    client = TestClient(app)
    return client, exporter, tracer


def test_middleware_creates_a_span_per_request() -> None:
    client, exporter, tracer = _make_app_with_span_capture()

    with patch("medagent.infra.middleware.tracer", tracer):
        response = client.get("/ping")

    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "GET /ping"
    assert span.attributes["http.method"] == "GET"
    assert span.attributes["http.target"] == "/ping"
    assert span.attributes["http.status_code"] == 200

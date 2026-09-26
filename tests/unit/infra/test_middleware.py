from unittest.mock import patch

import pytest
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
    assert span.attributes["http.route"] == "/ping"
    assert span.attributes["http.status_code"] == 200


# --- telemetry never carries the raw path or a patient id --------------------------------

import structlog  # noqa: E402

from medagent.infra.middleware import request_id_from  # noqa: E402
from tests.conftest import metric_value, span_text  # noqa: E402


def _app() -> TestClient:
    app = FastAPI()
    app.add_middleware(ObservabilityMiddleware)

    @app.get("/patients/{patient_id}/assess")
    async def assess(patient_id: str) -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_the_route_template_not_the_patient_id_labels_metrics_and_names_spans(spans) -> None:
    before = metric_value(
        "medagent_http_requests_total",
        method="GET",
        path="/patients/{patient_id}/assess",
        status_code="200",
    )

    _app().get("/patients/P-REAL-4711/assess")

    after = metric_value(
        "medagent_http_requests_total",
        method="GET",
        path="/patients/{patient_id}/assess",
        status_code="200",
    )
    assert after - before == 1
    span = next(s for s in spans.get_finished_spans() if s.name.startswith("GET "))
    assert span.name == "GET /patients/{patient_id}/assess"
    assert "p-real-4711" not in span_text(span)


def test_a_path_that_matches_no_route_collapses_to_unmatched(spans) -> None:
    """Scanners and typos would otherwise mint a new metric series per URL."""
    before = metric_value(
        "medagent_http_requests_total", method="GET", path="unmatched", status_code="404"
    )

    _app().get("/wp-admin/setup-config.php?x=zzz")

    after = metric_value(
        "medagent_http_requests_total", method="GET", path="unmatched", status_code="404"
    )
    assert after - before == 1
    span = next(s for s in spans.get_finished_spans() if s.name.startswith("GET "))
    assert span.name == "GET unmatched"
    assert "wp-admin" not in span_text(span)


def test_the_request_id_is_echoed_and_a_well_formed_inbound_one_is_kept() -> None:
    response = _app().get("/patients/P-1/assess", headers={"X-Request-ID": "caller-req-0001"})
    assert response.headers["x-request-id"] == "caller-req-0001"


def test_a_missing_request_id_is_generated() -> None:
    response = _app().get("/patients/P-1/assess")
    assert len(response.headers["x-request-id"]) == 32


@pytest.mark.parametrize(
    "hostile",
    ["short", "has spaces in it!!", "x" * 65, "line\nbreak-injection", "../../etc/passwd"],
)
def test_a_malformed_inbound_request_id_is_replaced(hostile: str) -> None:
    """The id is written into every log line, so it must not be able to smuggle
    content (newlines, oversized values) into them."""
    assert request_id_from(hostile) != hostile
    assert len(request_id_from(hostile)) == 32


def test_the_request_id_is_bound_for_the_log_lines_of_that_request() -> None:
    app = FastAPI()
    app.add_middleware(ObservabilityMiddleware)
    seen: dict[str, object] = {}

    @app.get("/probe")
    async def probe() -> dict[str, str]:
        seen.update(structlog.contextvars.get_contextvars())
        return {}

    TestClient(app).get("/probe", headers={"X-Request-ID": "bound-req-0002"})

    assert seen["request_id"] == "bound-req-0002"

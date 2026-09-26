"""ASGI middleware: request ids, structured request logging, metrics, and tracing.

Telemetry never carries the raw request path: /patients/P-123/assess would put a
patient identifier into Prometheus labels, span names and log lines (and one
series per patient). Everything uses the route template instead --
/patients/{patient_id}/assess -- and unmatched paths (scanners, typos) collapse
to "unmatched" so random URLs cannot create unbounded series.
"""

import re
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from medagent.infra.logging import get_logger
from medagent.infra.metrics import http_request_duration_seconds, http_requests_total
from medagent.infra.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
UNMATCHED_ROUTE = "unmatched"


def request_id_from(inbound: str | None) -> str:
    """Honours a well-formed inbound id (so a caller can correlate its own logs);
    anything else -- including a value that could smuggle content into logs --
    is replaced."""
    if inbound and _VALID_REQUEST_ID.match(inbound):
        return inbound
    return uuid.uuid4().hex


def route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or UNMATCHED_ROUTE


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request_id_from(request.headers.get(REQUEST_ID_HEADER))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # The route template is only known once routing has run, so the span is
        # named generically and renamed below.
        with tracer.start_as_current_span("HTTP") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("medagent.request_id", request_id)

            start = time.monotonic()
            response = await call_next(request)
            duration = time.monotonic() - start

            route = route_template(request)
            span.update_name(f"{request.method} {route}")
            span.set_attribute("http.route", route)
            span.set_attribute("http.status_code", response.status_code)

        http_requests_total.labels(
            method=request.method, path=route, status_code=str(response.status_code)
        ).inc()
        http_request_duration_seconds.labels(method=request.method, path=route).observe(duration)

        logger.info(
            "http_request",
            method=request.method,
            route=route,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

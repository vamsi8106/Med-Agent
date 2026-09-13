"""ASGI middleware: structured request logging, Prometheus metrics, and tracing."""

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from medagent.infra.logging import get_logger
from medagent.infra.metrics import http_request_duration_seconds, http_requests_total
from medagent.infra.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        with tracer.start_as_current_span(f"{request.method} {path}") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.target", path)

            start = time.monotonic()
            response = await call_next(request)
            duration = time.monotonic() - start

            span.set_attribute("http.status_code", response.status_code)

        http_requests_total.labels(
            method=request.method, path=path, status_code=str(response.status_code)
        ).inc()
        http_request_duration_seconds.labels(method=request.method, path=path).observe(duration)

        logger.info(
            "http_request",
            method=request.method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )
        return response

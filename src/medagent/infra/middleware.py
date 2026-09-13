"""ASGI middleware: structured request logging and Prometheus metrics."""

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from medagent.infra.logging import get_logger
from medagent.infra.metrics import http_request_duration_seconds, http_requests_total

logger = get_logger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        path = request.url.path
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

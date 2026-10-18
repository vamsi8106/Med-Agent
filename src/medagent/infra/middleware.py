"""ASGI middleware: request ids, structured request logging, metrics, tracing, and
request-size / security-header protection.

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
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    # Responses carry patient data: no browser or proxy may cache them.
    (b"cache-control", b"no-store"),
]
_HSTS = (b"strict-transport-security", b"max-age=31536000; includeSubDomains")


class SecurityMiddleware:
    """Pure ASGI (so it sees the raw byte stream): refuses request bodies over
    max_body_bytes with 413 -- by Content-Length up front, and by counting
    streamed chunks for requests without one -- and adds security headers to
    every HTTP response. HSTS only when TLS is in front (production)."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int, hsts: bool) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.headers = [*_SECURITY_HEADERS, *([_HSTS] if hsts else [])]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._too_large(send)
            return

        received = 0
        started = False
        refused = False

        async def send_with_headers(message: Message) -> None:
            nonlocal started
            if refused:
                return  # we already answered 413; drop whatever the app sends
            if message["type"] == "http.response.start":
                started = True
                existing = {name.lower() for name, _ in message.get("headers", [])}
                message["headers"] = [
                    *message.get("headers", []),
                    *((name, value) for name, value in self.headers if name not in existing),
                ]
            await send(message)

        async def counting_receive() -> Message:
            # Raising here would surface as FastAPI's generic 400 ("error parsing
            # the body"), so answer 413 directly and tell the app the client left.
            nonlocal received, refused
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    if not started and not refused:
                        await self._too_large(send)
                    refused = True
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, counting_receive, send_with_headers)

    async def _too_large(self, send: Send) -> None:
        body = b'{"detail":"Request body too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    *self.headers,
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

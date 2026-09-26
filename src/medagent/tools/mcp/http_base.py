"""Shared HTTP transport for containerized MCP-adjacent REST servers.

Replaces the old stdio subprocess transport (MCPStdioClient) now that each
MCP server runs as its own network-reachable container: healthcare-mcp and
med-research-mcp-suite ship their own REST APIs, and medical-mcp (stdio-only)
runs behind a stdio<->HTTP bridge sidecar. Same cross-cutting protections as
before -- circuit breaker, retry with backoff, per-instance rate limiting,
tracing -- just over httpx instead of a stdio ClientSession.
"""

import asyncio
from typing import Any, Self

import httpx
from opentelemetry.propagate import inject

from medagent.core.exceptions import MCPError, UpstreamError
from medagent.infra.circuit_breaker import CircuitBreaker
from medagent.infra.logging import get_logger
from medagent.infra.metrics import mcp_requests_total
from medagent.infra.rate_limiter import TokenBucket
from medagent.infra.retry import retry
from medagent.infra.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


def _status_error(response: httpx.Response, url: str) -> MCPError:
    """Classifies an HTTP error status: 5xx and 408 mean the server is unwell
    (retryable, count toward the breaker); 429 asks us to slow down; any other
    4xx means our request is wrong, so retrying can't help and it must not make
    a healthy server look broken."""
    message = f"Request to {url} returned {response.status_code}: {response.text}"
    status = response.status_code
    if status == 429:
        retry_after: float | None
        try:
            retry_after = float(response.headers["retry-after"])
        except (KeyError, ValueError):
            retry_after = None
        return MCPError(message, reason="rate_limited", retryable=True, retry_after=retry_after)
    if status == 408:
        return MCPError(message, reason="timeout", retryable=True)
    if status >= 500:
        return MCPError(message, reason="server_error", retryable=True)
    return MCPError(message, reason="client_error")


class HttpMCPClient:
    # Path of the server's own health endpoint, for the readiness probe.
    health_path = "/health"

    def __init__(
        self,
        base_url: str,
        circuit_breaker: CircuitBreaker | None = None,
        timeout_seconds: float = 30.0,
        rate_limit_per_second: float = 5.0,
        rate_limit_capacity: int = 10,
        *,
        name: str = "mcp",
        retry_max_wait_seconds: float = 10.0,
        retry_budget_seconds: float = 45.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._name = name
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        # One instance is shared by every request the app serves, so several
        # users can be inside `async with` at once. Count them: the httpx
        # client opens with the first and closes only when the last leaves.
        self._open_users = 0
        self._lifecycle = asyncio.Lock()
        self._breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0, name=name, error_type=MCPError
        )
        self._bucket = TokenBucket(rate_limit_per_second, rate_limit_capacity)
        self._request_with_retry = retry(
            max_attempts=3,
            max_wait=retry_max_wait_seconds,
            budget=retry_budget_seconds,
            target="mcp",
        )(self._request_once)

    async def __aenter__(self) -> Self:
        # Self, not HttpMCPClient -- callers using `async with SubclassClient()`
        # need the subclass's own extra methods visible on the bound name.
        async with self._lifecycle:
            if self._open_users == 0:
                self._client = httpx.AsyncClient(
                    base_url=self._base_url, timeout=self._timeout_seconds
                )
            self._open_users += 1
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        async with self._lifecycle:
            self._open_users -= 1
            if self._open_users == 0 and self._client is not None:
                await self._client.aclose()
                self._client = None

    @property
    def breaker_state(self) -> str:
        return self._breaker.state.value

    async def probe(self, timeout_seconds: float = 2.0) -> bool:
        """Readiness probe: does the server's health endpoint answer? Uses its own
        short-lived client, outside the breaker and rate limiter, so a probe never
        counts as a failure or consumes request budget."""
        async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout_seconds) as client:
            response = await client.get(self.health_path)
        return response.status_code < 500

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        with tracer.start_as_current_span(f"mcp_http.{method.lower()}.{path}") as span:
            span.set_attribute("mcp.base_url", self._base_url)
            span.set_attribute("mcp.path", path)
            span.set_attribute("medagent.mcp.server", self._name)
            try:
                result = await self._breaker.call(self._request_with_retry, method, path, **kwargs)
            except UpstreamError as exc:
                span.set_attribute("medagent.mcp.outcome", exc.reason)
                mcp_requests_total.labels(server=self._name, outcome=exc.reason).inc()
                raise
            span.set_attribute("medagent.mcp.outcome", "success")
            mcp_requests_total.labels(server=self._name, outcome="success").inc()
            return result

    async def _request_once(self, method: str, path: str, **kwargs: Any) -> Any:
        if self._client is None:
            raise MCPError("HTTP client is not open; use 'async with' before making requests")

        await self._bucket.acquire()
        url = f"{self._base_url}{path}"
        # Propagate the trace to the MCP server (W3C traceparent), so its side of
        # the call can join this trace. Headers only -- no request content.
        headers = dict(kwargs.pop("headers", None) or {})
        inject(headers)
        kwargs["headers"] = headers
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise MCPError(
                f"Request to {url} timed out after {self._timeout_seconds}s",
                reason="timeout",
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise MCPError(
                f"Request to {url} failed: {exc}", reason="connection_error", retryable=True
            ) from exc
        except Exception as exc:
            raise MCPError(f"Request to {url} failed: {exc}") from exc

        if response.status_code >= 400:
            raise _status_error(response, url)
        try:
            return response.json()
        except ValueError as exc:
            raise MCPError(f"Request to {url} returned invalid JSON") from exc

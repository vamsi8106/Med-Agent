"""Shared HTTP transport for containerized MCP-adjacent REST servers.

Replaces the old stdio subprocess transport (MCPStdioClient) now that each
MCP server runs as its own network-reachable container: healthcare-mcp and
med-research-mcp-suite ship their own REST APIs, and medical-mcp (stdio-only)
runs behind a stdio<->HTTP bridge sidecar. Same cross-cutting protections as
before -- circuit breaker, retry with backoff, per-instance rate limiting,
tracing -- just over httpx instead of a stdio ClientSession.
"""

from typing import Any

import httpx

from medagent.core.exceptions import MCPError
from medagent.infra.circuit_breaker import CircuitBreaker
from medagent.infra.logging import get_logger
from medagent.infra.rate_limiter import TokenBucket
from medagent.infra.retry import retry
from medagent.infra.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class HttpMCPClient:
    def __init__(
        self,
        base_url: str,
        circuit_breaker: CircuitBreaker | None = None,
        timeout_seconds: float = 30.0,
        rate_limit_per_second: float = 5.0,
        rate_limit_capacity: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0
        )
        self._bucket = TokenBucket(rate_limit_per_second, rate_limit_capacity)

    async def __aenter__(self) -> "HttpMCPClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout_seconds)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        with tracer.start_as_current_span(f"mcp_http.{method.lower()}.{path}") as span:
            span.set_attribute("mcp.base_url", self._base_url)
            span.set_attribute("mcp.path", path)
            return await self._breaker.call(self._request_with_retry, method, path, **kwargs)

    @retry(max_attempts=3, exceptions=(Exception,))
    async def _request_with_retry(self, method: str, path: str, **kwargs: Any) -> Any:
        if self._client is None:
            raise MCPError("HTTP client is not open; use 'async with' before making requests")

        await self._bucket.acquire()
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise MCPError(
                f"Request to {self._base_url}{path} timed out after {self._timeout_seconds}s"
            ) from exc
        except Exception as exc:
            raise MCPError(f"Request to {self._base_url}{path} failed: {exc}") from exc

        if response.status_code >= 400:
            raise MCPError(
                f"Request to {self._base_url}{path} returned {response.status_code}: "
                f"{response.text}"
            )
        return response.json()

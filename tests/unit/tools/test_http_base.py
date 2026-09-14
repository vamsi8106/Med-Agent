import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medagent.core.exceptions import MCPError
from medagent.infra.circuit_breaker import CircuitBreaker, CircuitState
from medagent.tools.mcp.http_base import HttpMCPClient


@asynccontextmanager
async def _opened_client(handler: object, **kwargs: object):
    client = HttpMCPClient("http://test-server", **kwargs)  # type: ignore[arg-type]
    client._client = httpx.AsyncClient(
        base_url="http://test-server",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        timeout=client._timeout_seconds,
    )
    try:
        yield client
    finally:
        await client._client.aclose()


def _ok_handler(json_body: dict[str, object]) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json_body)

    return handler


async def test_request_returns_json_body() -> None:
    async with _opened_client(_ok_handler({"content": [{"text": "ok"}]})) as client:
        result = await client.request("POST", "/call-tool", json={"name": "search-drugs"})

    assert result == {"content": [{"text": "ok"}]}


async def test_request_without_open_client_raises() -> None:
    client = HttpMCPClient("http://test-server")
    with pytest.raises(MCPError):
        await client.request("POST", "/call-tool", json={})


async def test_error_status_raises_mcp_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream down")

    async with _opened_client(handler) as client:
        with pytest.raises(MCPError):
            await client.request("POST", "/call-tool", json={})


async def test_circuit_breaker_opens_and_short_circuits_further_calls() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500, text="upstream down")

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)
    async with _opened_client(handler, circuit_breaker=breaker) as client:
        with pytest.raises(MCPError):
            await client.request("POST", "/call-tool", json={})
        assert breaker.state is CircuitState.OPEN

        calls_before = call_count
        with pytest.raises(MCPError, match="Circuit breaker is open"):
            await client.request("POST", "/call-tool", json={})
        assert call_count == calls_before


async def test_request_timeout_raises_mcp_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async with _opened_client(handler) as client:
        with pytest.raises(MCPError, match="timed out"):
            await client.request("POST", "/call-tool", json={})


async def test_request_acquires_rate_limit_token_per_attempt() -> None:
    async with _opened_client(_ok_handler({"ok": True})) as client:
        client._bucket.acquire = AsyncMock(wraps=client._bucket.acquire)
        await client.request("POST", "/call-tool", json={})
        client._bucket.acquire.assert_awaited_once()


async def test_rate_limit_throttles_burst_of_calls() -> None:
    async with _opened_client(
        _ok_handler({"ok": True}), rate_limit_per_second=5, rate_limit_capacity=1
    ) as client:
        start = asyncio.get_event_loop().time()
        await client.request("POST", "/call-tool", json={})
        await client.request("POST", "/call-tool", json={})
        elapsed = asyncio.get_event_loop().time() - start

    # capacity=1 at 5/s means the second call must wait ~0.2s for a token
    assert elapsed >= 0.15


async def test_request_creates_a_trace_span() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    import medagent.tools.mcp.http_base as http_base_module

    async with _opened_client(_ok_handler({"ok": True})) as client:
        original_tracer = http_base_module.tracer
        http_base_module.tracer = tracer
        try:
            await client.request("POST", "/call-tool", json={})
        finally:
            http_base_module.tracer = original_tracer

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "mcp_http.post./call-tool"
    assert spans[0].attributes["mcp.path"] == "/call-tool"

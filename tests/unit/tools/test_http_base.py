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
        with pytest.raises(MCPError, match="Circuit breaker for .* is open") as excinfo:
            await client.request("POST", "/call-tool", json={})
        assert excinfo.value.reason == "circuit_open"
        assert excinfo.value.retryable is False
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


# --- re-entrancy: one shared instance, overlapping users ------------------------


def _slow_ok_handler(delay: float) -> object:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay)
        return httpx.Response(200, json={"content": [{"text": "ok"}]})

    return handler


@pytest.fixture
def mocked_transport(monkeypatch: pytest.MonkeyPatch):
    """Makes every httpx.AsyncClient the client creates use a slow mock server,
    so the real __aenter__/__aexit__ lifecycle is exercised."""
    real = httpx.AsyncClient

    def factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real(*args, transport=httpx.MockTransport(_slow_ok_handler(0.05)), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("medagent.tools.mcp.http_base.httpx.AsyncClient", factory)


async def _use(client: HttpMCPClient, start_delay: float, hold: float) -> str:
    await asyncio.sleep(start_delay)
    async with client as c:
        await c.request("POST", "/call-tool", json={})
        await asyncio.sleep(hold)
        await c.request("POST", "/call-tool", json={})
    return "ok"


async def test_overlapping_users_of_one_shared_client_both_succeed(mocked_transport: None) -> None:
    """Regression: AppState shares one MCP client across requests. The first
    user to leave `async with` used to close the httpx client under the other,
    which then failed with "HTTP client is not open" -- so two doctors hitting
    the app at once could break each other's lookups."""
    shared = HttpMCPClient("http://test-server")

    results = await asyncio.gather(_use(shared, 0.0, 0.05), _use(shared, 0.02, 0.3))

    assert results == ["ok", "ok"]


async def test_nested_use_keeps_the_client_open_until_the_last_exit(mocked_transport: None) -> None:
    client = HttpMCPClient("http://test-server")

    async with client:
        async with client:
            pass
        # the inner exit must not have closed it for the outer user
        assert await client.request("POST", "/call-tool", json={}) == {"content": [{"text": "ok"}]}

    assert client._client is None


async def test_client_is_unusable_again_after_every_user_has_left(mocked_transport: None) -> None:
    client = HttpMCPClient("http://test-server")
    async with client:
        pass

    with pytest.raises(MCPError, match="not open"):
        await client.request("POST", "/call-tool", json={})


async def test_client_can_be_reopened_after_being_fully_closed(mocked_transport: None) -> None:
    client = HttpMCPClient("http://test-server")
    async with client:
        pass

    async with client:
        assert await client.request("POST", "/call-tool", json={})


# --- reliability: classification, retries, breaker -----------------------------------

from tests.conftest import FakeClock, metric_value  # noqa: E402


def _counting_handler(*responses: httpx.Response | Exception):
    """Plays a script (exceptions are raised); repeats the last when exhausted."""
    script = list(responses)
    count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        count["n"] += 1
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, Exception):
            raise item
        return item

    return handler, count


async def test_a_client_error_is_not_retried_and_does_not_trip_the_breaker(
    clock: FakeClock,
) -> None:
    handler, count = _counting_handler(httpx.Response(404, text="no such tool"))
    breaker = CircuitBreaker(failure_threshold=1, name="test-404")
    async with _opened_client(handler, circuit_breaker=breaker) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.request("POST", "/call-tool", json={})

    assert count["n"] == 1
    assert excinfo.value.reason == "client_error"
    assert breaker.state is CircuitState.CLOSED


async def test_a_server_error_is_retried_to_the_limit_and_counts_toward_the_breaker(
    clock: FakeClock,
) -> None:
    handler, count = _counting_handler(httpx.Response(503, text="overloaded"))
    breaker = CircuitBreaker(failure_threshold=1, name="test-503")
    async with _opened_client(handler, circuit_breaker=breaker) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.request("POST", "/call-tool", json={})

    assert count["n"] == 3
    assert excinfo.value.reason == "server_error"
    assert breaker.state is CircuitState.OPEN


async def test_a_connection_failure_is_retried(clock: FakeClock) -> None:
    handler, count = _counting_handler(httpx.ConnectError("refused"))
    async with _opened_client(handler) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.request("POST", "/call-tool", json={})

    assert count["n"] == 3
    assert excinfo.value.reason == "connection_error"


async def test_a_rate_limit_with_a_short_retry_after_is_waited_out(clock: FakeClock) -> None:
    handler, count = _counting_handler(
        httpx.Response(429, headers={"retry-after": "1"}),
        httpx.Response(200, json={"ok": True}),
    )
    async with _opened_client(handler) as client:
        result = await client.request("POST", "/call-tool", json={})

    assert result == {"ok": True}
    assert count["n"] == 2
    assert 1.0 <= clock.sleeps[0] <= 1.5


async def test_a_rate_limit_with_a_long_retry_after_fails_fast(clock: FakeClock) -> None:
    handler, count = _counting_handler(httpx.Response(429, headers={"retry-after": "300"}))
    async with _opened_client(handler, retry_max_wait_seconds=10.0) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.request("POST", "/call-tool", json={})

    assert count["n"] == 1
    assert clock.sleeps == []
    assert excinfo.value.reason == "rate_limited"


async def test_an_unparseable_body_is_a_clear_error_and_is_not_retried(
    clock: FakeClock,
) -> None:
    handler, count = _counting_handler(httpx.Response(200, text="<html>not json</html>"))
    async with _opened_client(handler) as client:
        with pytest.raises(MCPError, match="invalid JSON"):
            await client.request("POST", "/call-tool", json={})

    assert count["n"] == 1


async def test_outcomes_are_recorded_per_server(clock: FakeClock) -> None:
    ok_before = metric_value("medagent_mcp_requests_total", server="test-srv", outcome="success")
    bad_before = metric_value(
        "medagent_mcp_requests_total", server="test-srv", outcome="client_error"
    )
    good, _ = _counting_handler(httpx.Response(200, json={}))
    bad, _ = _counting_handler(httpx.Response(400, text="bad"))

    async with _opened_client(good, name="test-srv") as client:
        await client.request("POST", "/x", json={})
    async with _opened_client(bad, name="test-srv") as client:
        with pytest.raises(MCPError):
            await client.request("POST", "/x", json={})

    ok_after = metric_value("medagent_mcp_requests_total", server="test-srv", outcome="success")
    bad_after = metric_value(
        "medagent_mcp_requests_total", server="test-srv", outcome="client_error"
    )
    assert (ok_after - ok_before, bad_after - bad_before) == (1, 1)

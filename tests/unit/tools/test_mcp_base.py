import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medagent.core.exceptions import MCPError
from medagent.infra.circuit_breaker import CircuitBreaker, CircuitState
from medagent.tools.mcp.base import MCPStdioClient


class _FakeSession:
    def __init__(self) -> None:
        self.initialize = AsyncMock()
        self.call_tool = AsyncMock(
            return_value=SimpleNamespace(isError=False, content=[SimpleNamespace(text="ok")])
        )


@asynccontextmanager
async def _fake_stdio_client(_params: object):
    yield ("read", "write")


async def test_call_tool_returns_content() -> None:
    fake_session = _FakeSession()

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
    ):
        async with MCPStdioClient("echo", []) as client:
            content = await client.call_tool("search-drugs", {"query": "metformin"})

    assert content[0].text == "ok"
    fake_session.call_tool.assert_awaited_once_with("search-drugs", {"query": "metformin"})


async def test_call_tool_without_session_raises() -> None:
    client = MCPStdioClient("echo", [])
    with pytest.raises(MCPError):
        await client.call_tool("search-drugs", {"query": "metformin"})


async def test_call_tool_error_result_raises() -> None:
    fake_session = _FakeSession()
    fake_session.call_tool = AsyncMock(
        return_value=SimpleNamespace(isError=True, content="bad request")
    )

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
    ):
        async with MCPStdioClient("echo", []) as client:
            with pytest.raises(MCPError):
                await client.call_tool("search-drugs", {"query": "metformin"})


async def test_circuit_breaker_opens_and_short_circuits_further_calls() -> None:
    fake_session = _FakeSession()
    fake_session.call_tool = AsyncMock(
        return_value=SimpleNamespace(isError=True, content="upstream down")
    )

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60)

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
    ):
        async with MCPStdioClient("echo", [], circuit_breaker=breaker) as client:
            with pytest.raises(MCPError):
                await client.call_tool("search-drugs", {"query": "metformin"})
            assert breaker.state is CircuitState.OPEN

            calls_before = fake_session.call_tool.await_count
            with pytest.raises(MCPError, match="Circuit breaker is open"):
                await client.call_tool("search-drugs", {"query": "metformin"})
            assert fake_session.call_tool.await_count == calls_before


async def test_call_tool_timeout_raises_mcp_error() -> None:
    fake_session = _FakeSession()

    async def _hangs_forever(*_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(10)

    fake_session.call_tool = _hangs_forever

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
    ):
        async with MCPStdioClient("echo", [], timeout_seconds=0.05) as client:
            with pytest.raises(MCPError, match="timed out"):
                await client.call_tool("search-drugs", {"query": "metformin"})


async def test_connection_setup_timeout_raises_mcp_error() -> None:
    @asynccontextmanager
    async def _hanging_stdio_client(_params: object):
        await asyncio.sleep(10)
        yield ("read", "write")

    with patch("medagent.tools.mcp.base.stdio_client", _hanging_stdio_client):
        with pytest.raises(MCPError, match="timed out"):
            async with MCPStdioClient("echo", [], timeout_seconds=0.05):
                pass


async def test_call_tool_acquires_rate_limit_token_per_attempt() -> None:
    fake_session = _FakeSession()

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
    ):
        async with MCPStdioClient("echo", []) as client:
            client._bucket.acquire = AsyncMock(wraps=client._bucket.acquire)
            await client.call_tool("search-drugs", {"query": "metformin"})
            client._bucket.acquire.assert_awaited_once()


async def test_rate_limit_throttles_burst_of_calls() -> None:
    fake_session = _FakeSession()

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
    ):
        async with MCPStdioClient(
            "echo", [], rate_limit_per_second=5, rate_limit_capacity=1
        ) as client:
            start = asyncio.get_event_loop().time()
            await client.call_tool("search-drugs", {"query": "a"})
            await client.call_tool("search-drugs", {"query": "b"})
            elapsed = asyncio.get_event_loop().time() - start

    # capacity=1 at 5/s means the second call must wait ~0.2s for a token
    assert elapsed >= 0.15


async def test_call_tool_creates_a_trace_span() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    fake_session = _FakeSession()

    @asynccontextmanager
    async def _fake_client_session(_read: object, _write: object):
        yield fake_session

    with (
        patch("medagent.tools.mcp.base.stdio_client", _fake_stdio_client),
        patch("medagent.tools.mcp.base.ClientSession", _fake_client_session),
        patch("medagent.tools.mcp.base.tracer", tracer),
    ):
        async with MCPStdioClient("echo", []) as client:
            await client.call_tool("search-drugs", {"query": "metformin"})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "mcp.call_tool.search-drugs"
    assert spans[0].attributes["mcp.tool_name"] == "search-drugs"
    assert spans[0].attributes["mcp.command"] == "echo"

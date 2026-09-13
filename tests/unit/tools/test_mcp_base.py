from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from medagent.core.exceptions import MCPError
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

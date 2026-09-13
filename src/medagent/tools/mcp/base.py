"""Shared stdio transport for MCP clients (medical, healthcare, med-research)."""

import asyncio
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from medagent.core.exceptions import MCPError
from medagent.infra.circuit_breaker import CircuitBreaker
from medagent.infra.logging import get_logger
from medagent.infra.retry import retry

logger = get_logger(__name__)


class MCPStdioClient:
    """Manages a stdio-transport MCP server subprocess and its client session.

    Calls go through a CircuitBreaker: once a server fails repeatedly, further
    calls fail fast with MCPError instead of hanging/retrying against a
    server that's down, until the breaker's recovery timeout elapses. Both
    connection setup (spawning the subprocess) and each tool call are bounded
    by timeout_seconds so a hung server/process can't stall a request forever.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        circuit_breaker: CircuitBreaker | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._server_params = StdioServerParameters(command=command, args=args)
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._timeout_seconds = timeout_seconds
        self._breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=5, recovery_timeout=30.0
        )

    async def __aenter__(self) -> "MCPStdioClient":
        self._exit_stack = AsyncExitStack()
        try:
            read, write = await asyncio.wait_for(
                self._exit_stack.enter_async_context(stdio_client(self._server_params)),
                timeout=self._timeout_seconds,
            )
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self._session.initialize(), timeout=self._timeout_seconds)
        except TimeoutError as exc:
            await self._exit_stack.aclose()
            raise MCPError(
                f"MCP server startup timed out after {self._timeout_seconds}s "
                f"(command={self._server_params.command})"
            ) from exc
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._session = None
        self._exit_stack = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._breaker.call(self._call_tool_with_retry, name, arguments)

    @retry(max_attempts=3, exceptions=(Exception,))
    async def _call_tool_with_retry(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise MCPError("MCP session is not open; use 'async with' before calling tools")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments), timeout=self._timeout_seconds
            )
        except TimeoutError as exc:
            raise MCPError(
                f"MCP tool call '{name}' timed out after {self._timeout_seconds}s"
            ) from exc
        except Exception as exc:
            raise MCPError(f"MCP tool call '{name}' failed: {exc}") from exc

        if getattr(result, "isError", False):
            raise MCPError(f"MCP tool '{name}' returned an error: {result.content}")
        return result.content

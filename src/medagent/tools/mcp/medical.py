"""Client for medical-mcp (FDA/WHO/RxNorm/PubMed), reached via a stdio<->HTTP bridge.

medical-mcp is stdio-only (it registers only a StdioServerTransport). The
bridge image (docker/medical-mcp-bridge) installs it as a pinned npm package
and exposes it over HTTP: `POST /call-tool` with {"name": ..., "arguments":
...} returns {"content": [...]}, the MCP tool-call result shape.
"""

from typing import Any

from medagent.core.config import Settings, get_settings
from medagent.tools.mcp.http_base import HttpMCPClient


class MedicalMCPClient(HttpMCPClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            base_url=settings.medical_mcp_url,
            timeout_seconds=settings.mcp_timeout_seconds,
            rate_limit_per_second=settings.mcp_rate_limit_per_second,
            rate_limit_capacity=settings.mcp_rate_limit_capacity,
            name="medical",
            retry_max_wait_seconds=settings.retry_max_wait_seconds,
            retry_budget_seconds=settings.retry_budget_seconds,
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        response = await self.request(
            "POST", "/call-tool", json={"name": name, "arguments": arguments}
        )
        return response.get("content", response) if isinstance(response, dict) else response

    async def search_drugs(self, query: str) -> Any:
        return await self.call_tool("search-drugs", {"query": query})

    async def search_medical_literature(self, query: str, max_results: int = 10) -> Any:
        return await self.call_tool(
            "search-medical-literature", {"query": query, "max_results": max_results}
        )

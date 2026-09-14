"""Client for medical-mcp (FDA/WHO/RxNorm/PubMed), reached via a stdio<->HTTP bridge.

medical-mcp itself is stdio-only (confirmed against its packed build: it only
registers a StdioServerTransport, no HTTP option) and isn't vendored in this
repo -- it's pulled via `npx -y medical-mcp`. To run it as its own
network-reachable container, it sits behind a small bridge sidecar
(docker/medical-mcp-bridge) that spawns it over stdio internally and exposes
a generic `POST /call-tool` HTTP endpoint: body {"name": ..., "arguments":
...}, response {"content": [...]} mirroring the real MCP tool-call result
shape. Update this mapping if the chosen bridge tool's contract differs.
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
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        response = await self.request(
            "POST", "/call-tool", json={"name": name, "arguments": arguments}
        )
        return response.get("content", response) if isinstance(response, dict) else response

    async def search_drugs(self, query: str) -> Any:
        return await self.call_tool("search-drugs", {"query": query})

    async def get_drug_details(self, drug_name: str) -> Any:
        return await self.call_tool("get-drug-details", {"drug_name": drug_name})

    async def search_drug_nomenclature(self, query: str) -> Any:
        return await self.call_tool("search-drug-nomenclature", {"query": query})

    async def get_health_statistics(self, indicator: str, country: str | None = None) -> Any:
        arguments: dict[str, Any] = {"indicator": indicator}
        if country is not None:
            arguments["country"] = country
        return await self.call_tool("get-health-statistics", arguments)

    async def search_medical_literature(self, query: str, max_results: int = 10) -> Any:
        return await self.call_tool(
            "search-medical-literature", {"query": query, "max_results": max_results}
        )

    async def get_article_details(self, article_id: str) -> Any:
        return await self.call_tool("get-article-details", {"article_id": article_id})

    async def fda_drug_lookup(self, drug_name: str) -> Any:
        return await self.call_tool("fda_drug_lookup", {"drug_name": drug_name})

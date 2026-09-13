"""Client for medical-mcp: FDA / WHO / RxNorm / PubMed data."""

from typing import Any

from medagent.core.config import Settings, get_settings
from medagent.tools.mcp.base import MCPStdioClient


class MedicalMCPClient(MCPStdioClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            command=settings.medical_mcp_command,
            args=settings.medical_mcp_args,
            timeout_seconds=settings.mcp_timeout_seconds,
        )

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

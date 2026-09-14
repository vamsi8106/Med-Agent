"""Client for healthcare-mcp's HTTP server: ICD-10 terminology, trials, calculators.

Not real MCP-over-HTTP -- confirmed against the actual project's
server/http-server.js, which exposes a generic POST /mcp/call-tool catch-all
(and a handful of GET convenience routes) backed by handleCallTool()'s real
tool names: fda_drug_lookup, pubmed_search, medrxiv_search, calculate_bmi,
ncbi_bookshelf_search, extract_dicom_metadata, health_topics,
clinical_trials_search, lookup_icd_code, get_usage_stats,
get_all_usage_stats. Routed through the generic endpoint below rather than
the GET routes, since it covers every tool with one consistent call shape.
"""

from typing import Any

from medagent.core.config import Settings, get_settings
from medagent.tools.mcp.http_base import HttpMCPClient


class HealthcareMCPClient(HttpMCPClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            base_url=settings.healthcare_mcp_url,
            timeout_seconds=settings.mcp_timeout_seconds,
            rate_limit_per_second=settings.mcp_rate_limit_per_second,
            rate_limit_capacity=settings.mcp_rate_limit_capacity,
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self.request(
            "POST", "/mcp/call-tool", json={"name": name, "arguments": arguments}
        )

    async def clinical_trials_search(
        self, condition: str, status: str | None = None, max_results: int | None = None
    ) -> Any:
        arguments: dict[str, Any] = {"condition": condition}
        if status is not None:
            arguments["status"] = status
        if max_results is not None:
            arguments["max_results"] = max_results
        return await self.call_tool("clinical_trials_search", arguments)

    async def medical_terminology(self, code_or_term: str) -> Any:
        return await self.call_tool(
            "lookup_icd_code", {"code": code_or_term, "description": code_or_term}
        )

    async def medical_calculator(self, height_meters: float, weight_kg: float) -> Any:
        return await self.call_tool(
            "calculate_bmi", {"height_meters": height_meters, "weight_kg": weight_kg}
        )

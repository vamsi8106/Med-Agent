"""Client for med-research-mcp-suite's web-server.ts REST API (cross-database analysis).

Not real MCP-over-HTTP -- confirmed against the actual project's
src/web-server.ts, which is a plain Express REST API (POST /api/analysis/*,
/api/trials/*, /api/fda/*), not the MCP JSON-RPC protocol. Each method below
maps to its real REST route rather than a generic call_tool dispatch, since
web-server.ts has no such generic endpoint.
"""

from typing import Any

from medagent.core.config import Settings, get_settings
from medagent.tools.mcp.http_base import HttpMCPClient


class ResearchMCPClient(HttpMCPClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            base_url=settings.research_mcp_url,
            timeout_seconds=settings.mcp_timeout_seconds,
            rate_limit_per_second=settings.mcp_rate_limit_per_second,
            rate_limit_capacity=settings.mcp_rate_limit_capacity,
        )

    async def comprehensive_analysis(self, drug_name: str, condition: str) -> Any:
        return await self.request(
            "POST",
            "/api/analysis/comprehensive",
            json={"drugName": drug_name, "condition": condition},
        )

    async def drug_safety_profile(self, drug_name: str) -> Any:
        return await self.request("POST", "/api/analysis/safety", json={"drugName": drug_name})

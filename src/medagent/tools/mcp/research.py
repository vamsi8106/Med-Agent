"""Client for med-research-mcp-suite: cross-database analysis."""

from typing import Any

from medagent.core.config import Settings, get_settings
from medagent.tools.mcp.base import MCPStdioClient


class ResearchMCPClient(MCPStdioClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            command=settings.research_mcp_command,
            args=settings.research_mcp_args,
            timeout_seconds=settings.mcp_timeout_seconds,
            rate_limit_per_second=settings.mcp_rate_limit_per_second,
            rate_limit_capacity=settings.mcp_rate_limit_capacity,
        )

    async def comprehensive_analysis(self, drug_name: str, condition: str) -> Any:
        # Real tool name/schema confirmed against the actual project (AGENTS.md's
        # "comprehensive-analysis" / single drug_name never matched this server):
        # it takes a drug + a condition, not a drug pair. For interaction checks,
        # the second drug's name is passed as `condition` -- the server's
        # underlying literature/trials search surfaces interaction-relevant
        # results for that combination even though it isn't a dedicated
        # drug-drug interaction endpoint.
        return await self.call_tool(
            "research_comprehensive_analysis", {"drugName": drug_name, "condition": condition}
        )

    async def drug_safety_profile(self, drug_name: str) -> Any:
        return await self.call_tool("research_drug_safety_profile", {"drugName": drug_name})

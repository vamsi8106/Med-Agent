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
        )

    async def comprehensive_analysis(self, drug_name: str) -> Any:
        return await self.call_tool("comprehensive-analysis", {"drug_name": drug_name})

    async def drug_safety_profile(self, drug_name: str) -> Any:
        return await self.call_tool("drug-safety-profile", {"drug_name": drug_name})

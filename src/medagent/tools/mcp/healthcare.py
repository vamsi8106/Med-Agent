"""Client for healthcare-mcp: ICD-10 terminology, clinical trials, calculators."""

from typing import Any

from medagent.core.config import Settings, get_settings
from medagent.tools.mcp.base import MCPStdioClient


class HealthcareMCPClient(MCPStdioClient):
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        super().__init__(
            command=settings.healthcare_mcp_command,
            args=settings.healthcare_mcp_args,
            timeout_seconds=settings.mcp_timeout_seconds,
        )

    async def search_clinical_guidelines(self, query: str) -> Any:
        return await self.call_tool("search-clinical-guidelines", {"query": query})

    async def clinical_trials_search(
        self, condition: str, phase: str | None = None, status: str | None = None
    ) -> Any:
        arguments: dict[str, Any] = {"condition": condition}
        if phase is not None:
            arguments["phase"] = phase
        if status is not None:
            arguments["status"] = status
        return await self.call_tool("clinical_trials_search", arguments)

    async def medical_terminology(self, code_or_term: str) -> Any:
        return await self.call_tool("medical_terminology", {"code_or_term": code_or_term})

    async def medical_calculator(self, calculator_name: str, inputs: dict[str, Any]) -> Any:
        return await self.call_tool(
            "medical_calculator", {"calculator_name": calculator_name, "inputs": inputs}
        )

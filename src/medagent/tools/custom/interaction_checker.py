"""Pairwise drug interaction checks via the medical-mcp and research MCP servers."""

import re
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

from medagent.core.models import DrugInteraction, Medication
from medagent.core.types import InteractionSeverity
from medagent.infra.logging import get_logger
from medagent.tools.base import BaseTool, ToolResult
from medagent.tools.mcp.research import ResearchMCPClient

logger = get_logger(__name__)

_SEVERITY_KEYWORDS: dict[InteractionSeverity, tuple[str, ...]] = {
    InteractionSeverity.CONTRAINDICATED: ("contraindicated",),
    InteractionSeverity.MAJOR: ("major", "severe"),
    InteractionSeverity.MODERATE: ("moderate",),
    InteractionSeverity.MINOR: ("minor",),
}

# med-research-mcp-suite's real response format (confirmed against a live
# call) reports "## RISK PROFILE\n<High|Medium|Low>: ..." rather than the
# major/moderate/minor wording above.
_RISK_PROFILE_PATTERN = re.compile(r"risk profile\s*\n?\s*(high|medium|low)", re.IGNORECASE)
_RISK_PROFILE_TO_SEVERITY = {
    "high": InteractionSeverity.MAJOR,
    "medium": InteractionSeverity.MODERATE,
    "low": InteractionSeverity.MINOR,
}


def _parse_severity(text: str) -> InteractionSeverity:
    lowered = text.lower()
    for severity, keywords in _SEVERITY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return severity

    risk_match = _RISK_PROFILE_PATTERN.search(text)
    if risk_match:
        return _RISK_PROFILE_TO_SEVERITY[risk_match.group(1).lower()]

    return InteractionSeverity.NONE


def _extract_text(response: Any) -> str:
    """Flattens the research-mcp REST JSON response body into severity-parseable text.

    web-server.ts returns {"success": bool, "data": {...}, ...} rather than
    MCP content blocks, so this just stringifies the whole payload -- the
    severity/risk-profile wording _parse_severity looks for lives somewhere
    inside `data` regardless of its exact nested shape.
    """
    if isinstance(response, dict):
        return str(response.get("data", response))
    return str(response)


class InteractionCheckerTool(BaseTool):
    name = "interaction_checker"
    description = "Checks all pairwise drug interactions for a medication list via MCP."

    def __init__(self, research_client: ResearchMCPClient | None = None) -> None:
        self._research_client = research_client or ResearchMCPClient()

    async def run(self, medications: list[Medication]) -> ToolResult:
        try:
            interactions = await self._check_all_pairs(medications)
        except Exception as exc:  # noqa: BLE001 - boundary: convert to ToolResult
            return ToolResult(tool_name=self.name, success=False, error=str(exc))
        return ToolResult(tool_name=self.name, success=True, data=interactions)

    async def _check_all_pairs(self, medications: list[Medication]) -> list[DrugInteraction]:
        interactions: list[DrugInteraction] = []
        async with self._research_client as client:
            for drug_a, drug_b in combinations(medications, 2):
                content = await client.comprehensive_analysis(
                    drug_a.name, f"{drug_b.name} interaction"
                )
                text = _extract_text(content)
                interactions.append(
                    DrugInteraction(
                        drug_a=drug_a.name,
                        drug_b=drug_b.name,
                        severity=_parse_severity(text),
                        description=text,
                        source="med-research-mcp-suite",
                        checked_at=datetime.now(UTC),
                    )
                )
        return interactions

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

# med-research-mcp-suite's REST response nests a structured
# data.riskProfile.level field ("High"/"Medium"/"Low") -- this is the
# primary signal. The regex fallback below exists only for the older
# MCP-content markdown format ("## RISK PROFILE\n<High|Medium|Low>: ..."),
# kept in case any caller still surfaces that shape.
_RISK_PROFILE_PATTERN = re.compile(r"risk profile\s*\n?\s*(high|medium|low)", re.IGNORECASE)
_RISK_PROFILE_TO_SEVERITY = {
    "high": InteractionSeverity.MAJOR,
    "medium": InteractionSeverity.MODERATE,
    "low": InteractionSeverity.MINOR,
}


def _parse_severity(response: Any) -> InteractionSeverity:
    """Reads severity from the real REST response shape first.

    Bug fixed here: this used to stringify the whole response dict and
    regex-search the result for "risk profile\\nmedium" -- but Python's
    str(dict) renders as "'riskProfile': {'level': 'Medium'", which that
    regex can never match (confirmed against a live call: every case with
    real risk data silently fell through to NONE). Read the structured
    field directly instead; the keyword/regex checks remain as fallbacks
    for response shapes with no riskProfile dict at all.
    """
    data = response.get("data", response) if isinstance(response, dict) else response
    if isinstance(data, dict):
        risk_profile = data.get("riskProfile")
        if isinstance(risk_profile, dict):
            level = str(risk_profile.get("level", "")).lower()
            if level in _RISK_PROFILE_TO_SEVERITY:
                return _RISK_PROFILE_TO_SEVERITY[level]

    text = str(data).lower()
    for severity, keywords in _SEVERITY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
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
                interactions.append(
                    DrugInteraction(
                        drug_a=drug_a.name,
                        drug_b=drug_b.name,
                        severity=_parse_severity(content),
                        description=_extract_text(content),
                        source="med-research-mcp-suite",
                        checked_at=datetime.now(UTC),
                    )
                )
        return interactions

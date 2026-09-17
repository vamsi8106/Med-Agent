from typing import Any
from unittest.mock import AsyncMock

from medagent.core.models import Medication
from medagent.core.types import InteractionSeverity
from medagent.tools.custom.interaction_checker import InteractionCheckerTool


class _FakeResearchClient:
    def __init__(self, response_data: Any) -> None:
        self.comprehensive_analysis = AsyncMock(
            return_value={"success": True, "data": response_data}
        )

    async def __aenter__(self) -> "_FakeResearchClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


async def test_interaction_checker_flags_major_severity() -> None:
    fake_client = _FakeResearchClient("This is a major interaction risk.")
    checker = InteractionCheckerTool(research_client=fake_client)  # type: ignore[arg-type]

    result = await checker.run([Medication(name="Metformin"), Medication(name="Glimepiride")])

    assert result.success is True
    interactions = result.data
    assert len(interactions) == 1
    assert interactions[0].severity == InteractionSeverity.MAJOR
    assert interactions[0].drug_a == "Metformin"
    assert interactions[0].drug_b == "Glimepiride"


async def test_interaction_checker_parses_risk_profile_format() -> None:
    # Fallback path only: older MCP-content markdown text shape, kept in
    # case any caller still surfaces it. The real REST shape is covered by
    # test_interaction_checker_parses_structured_risk_profile below.
    fake_client = _FakeResearchClient("## RISK PROFILE\nMedium: No factors identified")
    checker = InteractionCheckerTool(research_client=fake_client)  # type: ignore[arg-type]

    result = await checker.run([Medication(name="Metformin"), Medication(name="Glimepiride")])

    assert result.data[0].severity == InteractionSeverity.MODERATE


async def test_interaction_checker_parses_structured_risk_profile() -> None:
    """Regression test for a real bug: med-research-mcp-suite's actual REST
    response nests data.riskProfile.level, e.g. {"level": "Medium"} -- the
    old stringify-then-regex approach could never match this shape (Python's
    str(dict) renders "'riskProfile': {'level': 'Medium'", not "risk
    profile\\nmedium"), so every real call silently fell through to NONE.
    """
    fake_client = _FakeResearchClient(
        {
            "executiveSummary": "Basic analysis available.",
            "riskProfile": {"level": "Medium", "factors": []},
        }
    )
    checker = InteractionCheckerTool(research_client=fake_client)  # type: ignore[arg-type]

    result = await checker.run([Medication(name="Metformin"), Medication(name="Glimepiride")])

    assert result.data[0].severity == InteractionSeverity.MODERATE


async def test_interaction_checker_no_pairs_for_single_drug() -> None:
    fake_client = _FakeResearchClient("n/a")
    checker = InteractionCheckerTool(research_client=fake_client)  # type: ignore[arg-type]

    result = await checker.run([Medication(name="Metformin")])

    assert result.success is True
    assert result.data == []

from datetime import UTC, datetime

import pytest

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.core.exceptions import ToolError
from medagent.core.models import DrugInteraction, Medication, PatientContext
from medagent.core.types import InteractionSeverity
from medagent.llm.mock_provider import MockLLMProvider
from medagent.tools.base import ToolResult


class _FakeInteractionChecker:
    def __init__(self, interactions: list[DrugInteraction]) -> None:
        self._interactions = interactions

    async def run(self, medications: list) -> ToolResult:
        return ToolResult(tool_name="interaction_checker", success=True, data=self._interactions)


def _patient() -> PatientContext:
    return PatientContext(id="P-TEST-001", name="Patient Alpha", age=61, sex="M")


async def test_gate_check_interactions_for_metformin_and_glimepiride() -> None:
    interactions = [
        DrugInteraction(
            drug_a="Metformin",
            drug_b="Glimepiride",
            severity=InteractionSeverity.MODERATE,
            description="Increased risk of hypoglycemia when combined.",
            source="med-research-mcp-suite",
            checked_at=datetime.now(UTC),
        )
    ]
    agent = DrugSafetyAgent(
        llm=MockLLMProvider(fixed_response="These drugs together raise hypoglycemia risk."),
        interaction_checker=_FakeInteractionChecker(interactions),  # type: ignore[arg-type]
    )

    result = await agent.run(_patient(), "Check interactions for Metformin + Glimepiride")

    assert "hypoglycemia" in result.lower()
    assert "Metformin + Glimepiride" in result
    assert "med-research-mcp-suite" in result
    assert "moderate" in result


async def test_single_drug_raises_tool_error() -> None:
    agent = DrugSafetyAgent(
        llm=MockLLMProvider(),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )

    with pytest.raises(ToolError):
        await agent.run(_patient(), "Check interactions for Metformin")


async def test_uses_known_medications_from_context() -> None:
    interactions: list[DrugInteraction] = []
    agent = DrugSafetyAgent(
        llm=MockLLMProvider(fixed_response="No concerns found."),
        interaction_checker=_FakeInteractionChecker(interactions),  # type: ignore[arg-type]
    )
    context = _patient()
    context.medications = [Medication(name="Lisinopril"), Medication(name="Losartan")]

    result = await agent.run(context, "any interactions to worry about?")

    assert "No interactions found" in result

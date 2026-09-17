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


async def test_run_result_returns_agent_result_with_interactions() -> None:
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
        llm=MockLLMProvider(fixed_response="Findings."),
        interaction_checker=_FakeInteractionChecker(interactions),  # type: ignore[arg-type]
    )

    result = await agent.run_result(_patient(), "Check interactions for Metformin + Glimepiride")

    assert result.role.value == "drug_safety"
    assert result.interactions == interactions


async def test_flags_allergy_conflict_with_checked_drug() -> None:
    agent = DrugSafetyAgent(
        llm=MockLLMProvider(fixed_response="Findings."),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )
    context = _patient()
    context.allergies = ["Sulfa"]

    result = await agent.run(context, "Check interactions for Sulfamethoxazole and Trimethoprim")

    assert "ALLERGY CONFLICT" in result
    assert "Sulfa" in result


async def test_no_allergy_conflict_line_when_no_match() -> None:
    agent = DrugSafetyAgent(
        llm=MockLLMProvider(fixed_response="Findings."),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )
    context = _patient()
    context.allergies = ["Penicillin"]

    result = await agent.run(context, "Check interactions for Metformin and Glimepiride")

    assert "ALLERGY CONFLICT" not in result


async def test_allergy_conflict_misses_cross_reactive_drug_class() -> None:
    """Known limitation, tracked deliberately: _check_allergy_conflicts is a
    case-insensitive substring match between the allergy string and the drug
    name -- it has no concept of drug classes. A patient allergic to
    Penicillin prescribed Amoxicillin (both beta-lactams; well-documented
    cross-reactivity) gets no conflict flagged at all, silently, because
    "penicillin" is not a substring of "amoxicillin" or vice versa.

    This test exists to make that gap visible in test output rather than
    have it discovered in production -- fixing it needs a real allergy/drug
    class lookup (e.g. an MCP-backed drug-class table), not a string tweak.
    """
    agent = DrugSafetyAgent(
        llm=MockLLMProvider(fixed_response="Findings."),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )
    context = _patient()
    context.allergies = ["Penicillin"]

    result = await agent.run(context, "Check interactions for Amoxicillin and Ibuprofen")

    # This SHOULD flag a conflict (beta-lactam cross-reactivity) but doesn't --
    # asserting the current (unsafe) behavior so a future fix has to update
    # this test, forcing a deliberate decision rather than an accidental one.
    assert "ALLERGY CONFLICT" not in result

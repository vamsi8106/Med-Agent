from medagent.agents.triage_agent import TriageAgent
from medagent.core.models import PatientContext
from medagent.core.types import AgentRole


def _patient() -> PatientContext:
    return PatientContext(id="P-TEST-001", name="Patient Alpha", age=50, sex="F")


async def test_routes_drug_interaction_message() -> None:
    agent = TriageAgent()
    roles = await agent.route(_patient(), "Check interactions for Metformin and Glimepiride")
    assert AgentRole.DRUG_SAFETY in roles


async def test_routes_trial_message() -> None:
    agent = TriageAgent()
    roles = await agent.route(_patient(), "Any clinical trials for this condition?")
    assert AgentRole.TRIAL_FINDER in roles


async def test_defaults_to_evidence_when_unmatched() -> None:
    agent = TriageAgent()
    roles = await agent.route(_patient(), "What should I do?")
    assert roles == [AgentRole.EVIDENCE]


async def test_run_returns_comma_joined_roles() -> None:
    agent = TriageAgent()
    result = await agent.run(_patient(), "Check interactions for Metformin and Glimepiride")
    assert "drug_safety" in result

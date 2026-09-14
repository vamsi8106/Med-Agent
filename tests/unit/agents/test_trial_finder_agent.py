from unittest.mock import AsyncMock

from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.core.models import PatientContext
from medagent.llm.mock_provider import MockLLMProvider


class _FakeHealthcareClient:
    def __init__(self, text: str) -> None:
        self.clinical_trials_search = AsyncMock(return_value={"data": text})

    async def __aenter__(self) -> "_FakeHealthcareClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _patient() -> PatientContext:
    context = PatientContext(id="P-TEST-001", name="Patient Alpha", age=55, sex="M")
    context.conditions = ["type 2 diabetes"]
    return context


async def test_find_trials_uses_patient_condition() -> None:
    healthcare_client = _FakeHealthcareClient("NCT12345: Metformin extended trial.")
    agent = TrialFinderAgent(
        llm=MockLLMProvider(fixed_response="One relevant trial found."),
        healthcare_client=healthcare_client,  # type: ignore[arg-type]
    )

    result = await agent.find_trials(_patient(), "any trials?")

    healthcare_client.clinical_trials_search.assert_awaited_once_with("type 2 diabetes")
    assert result.role.value == "trial_finder"
    assert "ClinicalTrials.gov" in result.summary

from unittest.mock import AsyncMock

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.core.models import ClinicalEvidence, PatientContext
from medagent.llm.mock_provider import MockLLMProvider


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMedicalClient:
    def __init__(self, text: str) -> None:
        self.search_medical_literature = AsyncMock(return_value=[_Block(text)])

    async def __aenter__(self) -> "_FakeMedicalClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _patient() -> PatientContext:
    return PatientContext(id="P-TEST-001", name="Patient Alpha", age=55, sex="M")


async def test_gather_evidence_combines_guidelines_and_literature() -> None:
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = [
        ClinicalEvidence(title="ADA Guideline", summary="...", source="ada-2024")
    ]
    medical_client = _FakeMedicalClient("Metformin reduces HbA1c by 1-2%.")

    agent = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="Metformin is effective first-line therapy."),
        medical_client=medical_client,  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )

    result = await agent.gather_evidence(_patient(), "first-line therapy for type 2 diabetes")

    assert result.role.value == "evidence"
    assert len(result.evidence) == 2
    assert "ADA Guideline" in result.summary
    assert "PubMed" in result.summary


async def test_run_returns_summary_string() -> None:
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = _FakeMedicalClient("n/a")

    agent = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="Summary text"),
        medical_client=medical_client,  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )

    result = await agent.run(_patient(), "any evidence?")
    assert "Summary text" in result

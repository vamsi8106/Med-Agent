from datetime import UTC, datetime
from unittest.mock import AsyncMock

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import ClinicalEvidence, LLMResponse, Message, PatientContext, Visit
from medagent.llm.mock_provider import MockLLMProvider


class _RecordingLLMProvider(BaseLLMProvider):
    def __init__(self, fixed_response: str = "mock response") -> None:
        self._fixed_response = fixed_response
        self.received_messages: list[Message] = []

    async def complete(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> LLMResponse:
        self.received_messages = messages
        return LLMResponse(content=self._fixed_response, model="mock-model")


class _FakeMedicalClient:
    def __init__(self, text: str) -> None:
        self.search_medical_literature = AsyncMock(return_value=[{"text": text}])

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


async def test_synthesis_prompt_includes_conditions_and_allergies() -> None:
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = _FakeMedicalClient("n/a")
    llm = _RecordingLLMProvider(fixed_response="n/a")

    agent = EvidenceAgent(
        llm=llm,
        medical_client=medical_client,  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )
    patient = PatientContext(
        id="P-TEST-002",
        name="Patient Beta",
        age=60,
        sex="F",
        conditions=["type 2 diabetes"],
        allergies=["Penicillin"],
    )

    await agent.gather_evidence(patient, "first-line therapy?")

    user_message = next(m for m in llm.received_messages if m.role == "user")
    assert "type 2 diabetes" in user_message.content
    assert "Penicillin" in user_message.content


async def test_synthesis_prompt_omits_preamble_when_no_record_data() -> None:
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = _FakeMedicalClient("n/a")
    llm = _RecordingLLMProvider(fixed_response="n/a")

    agent = EvidenceAgent(
        llm=llm,
        medical_client=medical_client,  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )

    await agent.gather_evidence(_patient(), "first-line therapy?")

    user_message = next(m for m in llm.received_messages if m.role == "user")
    assert "Patient context:" not in user_message.content


async def test_synthesis_prompt_includes_recent_visit_history() -> None:
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = _FakeMedicalClient("n/a")
    llm = _RecordingLLMProvider(fixed_response="n/a")

    agent = EvidenceAgent(
        llm=llm,
        medical_client=medical_client,  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )
    patient = PatientContext(
        id="P-TEST-003",
        name="Patient Gamma",
        age=50,
        sex="M",
        visits=[
            Visit(
                visit_date=datetime.now(UTC),
                chief_complaint="fatigue",
                assessment="Suspected anemia, ordered CBC.",
            )
        ],
    )

    await agent.gather_evidence(patient, "any evidence for iron supplementation?")

    user_message = next(m for m in llm.received_messages if m.role == "user")
    assert "fatigue" in user_message.content
    assert "Suspected anemia" in user_message.content

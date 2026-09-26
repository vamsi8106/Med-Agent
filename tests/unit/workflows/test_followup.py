from unittest.mock import AsyncMock

import pytest

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import PatientNotFoundError
from medagent.core.models import PatientContext
from medagent.llm.mock_provider import MockLLMProvider
from medagent.workflows.followup import run_followup


async def test_followup_raises_for_unknown_patient() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = None

    with pytest.raises(PatientNotFoundError):
        await run_followup(
            TriageAgent(),
            AsyncMock(),
            AsyncMock(),
            AsyncMock(),
            ReportAgent(),
            memory,
            "P-UNKNOWN",
            "hi",
        )


async def test_followup_recalls_and_runs_assessment() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-600", name="Patient Alpha", age=50, sex="F"
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = AsyncMock()
    medical_client.__aenter__.return_value = medical_client
    medical_client.search_medical_literature.return_value = "n/a"
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="Stable, continue current plan."),
        medical_client=medical_client,
        guideline_retriever=guideline_retriever,
    )

    report, patient = await run_followup(
        TriageAgent(),
        AsyncMock(),
        evidence,
        AsyncMock(),
        ReportAgent(),
        memory,
        "P-TEST-600",
        "follow-up visit",
    )

    assert "Patient Alpha" in report
    assert patient.id == "P-TEST-600"
    memory.save_patient.assert_not_awaited()


async def test_followup_forwards_doctor_id_to_memory() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-601", name="Patient Beta", age=45, sex="M", doctor_id="DR-TEST-001"
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = AsyncMock()
    medical_client.__aenter__.return_value = medical_client
    medical_client.search_medical_literature.return_value = "n/a"
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="n/a"),
        medical_client=medical_client,
        guideline_retriever=guideline_retriever,
    )

    await run_followup(
        TriageAgent(),
        AsyncMock(),
        evidence,
        AsyncMock(),
        ReportAgent(),
        memory,
        "P-TEST-601",
        "follow-up visit",
        doctor_id="DR-TEST-001",
    )

    memory.get_patient.assert_awaited_once_with("P-TEST-601", "DR-TEST-001")

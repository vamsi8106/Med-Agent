from unittest.mock import AsyncMock

import pytest

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import MedAgentError
from medagent.core.models import PatientContext
from medagent.llm.mock_provider import MockLLMProvider
from medagent.workflows.followup import run_followup


async def test_followup_raises_for_unknown_patient() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = None

    with pytest.raises(MedAgentError):
        await run_followup(
            TriageAgent(), AsyncMock(), AsyncMock(), ReportAgent(), memory, "P-UNKNOWN", "hi"
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

    result = await run_followup(
        TriageAgent(), AsyncMock(), evidence, ReportAgent(), memory, "P-TEST-600", "follow-up visit"
    )

    assert "Patient Alpha" in result
    memory.save_patient.assert_awaited_once()

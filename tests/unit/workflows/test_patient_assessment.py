from datetime import UTC, datetime
from unittest.mock import AsyncMock

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.models import (
    ClinicalEvidence,
    DrugInteraction,
    Medication,
    PatientContext,
)
from medagent.core.types import InteractionSeverity
from medagent.llm.mock_provider import MockLLMProvider
from medagent.tools.base import ToolResult
from medagent.workflows.patient_assessment import run_patient_assessment


class _FakeInteractionChecker:
    def __init__(self, interactions: list[DrugInteraction]) -> None:
        self._interactions = interactions

    async def run(self, medications: list) -> ToolResult:
        return ToolResult(tool_name="interaction_checker", success=True, data=self._interactions)


class _FakeMedicalClient:
    def __init__(self, text: str) -> None:
        self.search_medical_literature = AsyncMock(return_value=text)

    async def __aenter__(self) -> "_FakeMedicalClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _complex_patient() -> PatientContext:
    return PatientContext(
        id="P-TEST-500",
        name="Patient Alpha",
        age=68,
        sex="F",
        conditions=["type 2 diabetes", "hypertension"],
        medications=[Medication(name="Metformin"), Medication(name="Glimepiride")],
    )


async def test_gate_complex_patient_multi_agent_structured_report() -> None:
    interactions = [
        DrugInteraction(
            drug_a="Metformin",
            drug_b="Glimepiride",
            severity=InteractionSeverity.MODERATE,
            description="Increased hypoglycemia risk.",
            source="med-research-mcp-suite",
            checked_at=datetime.now(UTC),
        )
    ]
    triage = TriageAgent()
    drug_safety = DrugSafetyAgent(
        llm=MockLLMProvider(fixed_response="Monitor for hypoglycemia."),
        interaction_checker=_FakeInteractionChecker(interactions),  # type: ignore[arg-type]
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = [
        ClinicalEvidence(title="ADA Guideline", summary="...", source="ada-2024")
    ]
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="Metformin remains first-line."),
        medical_client=_FakeMedicalClient("PubMed article summary."),  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )
    report = ReportAgent()
    memory = AsyncMock()

    patient = _complex_patient()
    result = await run_patient_assessment(
        triage,
        drug_safety,
        evidence,
        report,
        memory,
        patient,
        "Check interactions for current medications and any relevant treatment evidence",
    )

    assert "Patient Alpha" in result
    assert "## Drug Safety" in result
    assert "## Evidence" in result
    assert "Metformin + Glimepiride" in result
    assert "ADA Guideline" in result
    memory.save_patient.assert_awaited_once_with(patient)


async def test_workflow_skips_drug_safety_when_single_medication() -> None:
    triage = TriageAgent()
    drug_safety = DrugSafetyAgent(
        llm=MockLLMProvider(),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="No major findings."),
        medical_client=_FakeMedicalClient("n/a"),  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )
    report = ReportAgent()
    memory = AsyncMock()

    patient = PatientContext(id="P-TEST-501", name="Patient Beta", age=40, sex="M")
    patient.medications = [Medication(name="Lisinopril")]

    result = await run_patient_assessment(
        triage, drug_safety, evidence, report, memory, patient, "any guidance?"
    )

    assert "## Drug Safety" not in result
    assert "Patient Beta" in result

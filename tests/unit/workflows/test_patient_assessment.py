from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import AllAgentsFailedError, MCPError, ProviderError, ToolError
from medagent.core.models import (
    AgentResult,
    ClinicalEvidence,
    DrugInteraction,
    LLMResponse,
    Medication,
    PatientContext,
)
from medagent.core.types import AgentRole, InteractionSeverity
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
    trial_finder = AsyncMock()
    report = ReportAgent()

    patient = _complex_patient()
    result = await run_patient_assessment(
        triage,
        drug_safety,
        evidence,
        trial_finder,
        report,
        patient,
        "Check interactions for current medications and any relevant treatment evidence",
    )
    trial_finder.find_trials.assert_not_awaited()

    assert "Patient Alpha" in result
    assert "## Drug Safety" in result
    assert "## Evidence" in result
    assert "Metformin + Glimepiride" in result
    assert "ADA Guideline" in result


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
    trial_finder = AsyncMock()
    report = ReportAgent()

    patient = PatientContext(id="P-TEST-501", name="Patient Beta", age=40, sex="M")
    patient.medications = [Medication(name="Lisinopril")]

    result = await run_patient_assessment(
        triage, drug_safety, evidence, trial_finder, report, patient, "any guidance?"
    )

    assert "## Drug Safety" not in result
    assert "Patient Beta" in result


async def test_workflow_skips_remaining_agents_once_token_budget_exhausted() -> None:
    class _UsageLLM(MockLLMProvider):
        async def complete(self, messages: list, tools: list | None = None) -> LLMResponse:  # type: ignore[override]
            return LLMResponse(
                content=self._fixed_response,
                model="mock-model",
                usage={"prompt_tokens": 100, "completion_tokens": 0},
            )

    triage = TriageAgent()
    drug_safety = DrugSafetyAgent(
        llm=_UsageLLM(fixed_response="Monitor for hypoglycemia."),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    evidence = EvidenceAgent(
        llm=_UsageLLM(fixed_response="Metformin remains first-line."),
        medical_client=_FakeMedicalClient("PubMed article summary."),  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )
    trial_finder = AsyncMock()
    report = ReportAgent()

    patient = _complex_patient()
    result = await run_patient_assessment(
        triage,
        drug_safety,
        evidence,
        trial_finder,
        report,
        patient,
        "Check interactions for current medications and any relevant treatment evidence",
        max_tokens=100,
    )

    assert "## Note" in result
    assert "token budget" in result
    assert "evidence" in result


async def test_workflow_runs_trial_finder_when_message_mentions_trials() -> None:
    triage = TriageAgent()
    drug_safety = DrugSafetyAgent(
        llm=MockLLMProvider(),
        interaction_checker=_FakeInteractionChecker([]),  # type: ignore[arg-type]
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="n/a"),
        medical_client=_FakeMedicalClient("n/a"),  # type: ignore[arg-type]
        guideline_retriever=guideline_retriever,
    )
    trial_finder = AsyncMock()
    trial_finder.find_trials.return_value = AgentResult(
        role=AgentRole.TRIAL_FINDER,
        summary="Found one relevant trial.",
        evidence=[
            ClinicalEvidence(
                title="Clinical trials for type 2 diabetes",
                summary="NCT12345",
                source="ClinicalTrials.gov",
            )
        ],
    )
    report = ReportAgent()

    patient = PatientContext(id="P-TEST-502", name="Patient Gamma", age=50, sex="F")

    result = await run_patient_assessment(
        triage, drug_safety, evidence, trial_finder, report, patient, "any clinical trials?"
    )

    trial_finder.find_trials.assert_awaited_once()
    assert "## Trial Finder" in result
    assert "Clinical trials for type 2 diabetes" in result


# --- partial failure -----------------------------------------------------------


def _agent_result(role: AgentRole, summary: str) -> AgentResult:
    return AgentResult(role=role, summary=summary)


async def _assess_with(drug_safety: object, evidence: object, trial_finder: object, message: str):
    return await run_patient_assessment(
        TriageAgent(),
        drug_safety,  # type: ignore[arg-type]
        evidence,  # type: ignore[arg-type]
        trial_finder,  # type: ignore[arg-type]
        ReportAgent(),
        _complex_patient(),
        message,
    )


_BOTH = "Check interactions for current medications and any relevant treatment evidence"


async def test_one_failed_specialist_still_yields_a_report_with_a_note() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "Interaction found.")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError("Groq 413 ... org_01SECRET123")

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "Interaction found." in report
    assert "## Note" in report
    assert "Unavailable: Evidence (the language model service was unavailable)" in report


async def test_failure_note_never_leaks_the_raw_exception_message() -> None:
    """Provider errors can embed account identifiers (the Groq 413 we hit
    contained the organisation id); only fixed, doctor-safe text may reach a
    report."""
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError("org_01SECRET123 tokens 9644")

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "org_01SECRET123" not in report
    assert "9644" not in report


async def test_failed_specialist_does_not_drop_the_deterministic_sections() -> None:
    patient = _complex_patient()
    patient.allergies = ["Sulfa"]
    drug_safety = AsyncMock()
    drug_safety.run_result.side_effect = MCPError("down")
    evidence = AsyncMock()
    evidence.gather_evidence.return_value = _agent_result(AgentRole.EVIDENCE, "Evidence text.")

    report = await run_patient_assessment(
        TriageAgent(), drug_safety, evidence, AsyncMock(), ReportAgent(), patient, _BOTH
    )

    assert "Evidence text." in report
    assert "## Known Allergies" in report
    assert "Unavailable: Drug Safety (an external medical data source was unavailable)" in report


async def test_a_tool_error_gets_the_generic_failure_text() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.side_effect = ToolError("At least two medications required")
    evidence = AsyncMock()
    evidence.gather_evidence.return_value = _agent_result(AgentRole.EVIDENCE, "fine")

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "Unavailable: Drug Safety (the check could not be completed)" in report


async def test_every_specialist_failing_raises_instead_of_returning_an_empty_report() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.side_effect = ProviderError("down")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = MCPError("down")

    with pytest.raises(AllAgentsFailedError) as excinfo:
        await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "drug_safety" in str(excinfo.value)
    assert "evidence" in str(excinfo.value)


async def test_an_unexpected_exception_is_a_bug_and_still_propagates() -> None:
    """Only MedAgentError is an expected operational failure. Swallowing a
    KeyError as "unavailable" would hide real bugs behind a friendly note."""
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = KeyError("bug")

    with pytest.raises(KeyError):
        await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)


async def test_failure_and_budget_skip_share_one_note() -> None:
    class _CostlyDrugSafety:
        async def run_result(self, context: PatientContext, message: str) -> AgentResult:
            return AgentResult(
                role=AgentRole.DRUG_SAFETY, summary="ok", usage={"prompt_tokens": 500}
            )

    evidence = AsyncMock()
    evidence.gather_evidence.return_value = _agent_result(AgentRole.EVIDENCE, "never runs")
    trial_finder = AsyncMock()
    trial_finder.find_trials.side_effect = ProviderError("down")

    report = await run_patient_assessment(
        TriageAgent(),
        _CostlyDrugSafety(),  # type: ignore[arg-type]
        evidence,
        trial_finder,
        ReportAgent(),
        _complex_patient(),
        f"{_BOTH} and any clinical trials",
        max_tokens=100,
    )

    assert report.count("## Note") == 1


# --- metrics and specific failure wording ---------------------------------------------

from medagent.agents.drug_safety_agent import DrugSafetyAgent as _RealDrugSafety  # noqa: E402
from medagent.tools.base import ToolResult as _ToolResult  # noqa: E402
from tests.conftest import metric_value  # noqa: E402


class _McpDownChecker:
    async def run(self, medications: list) -> _ToolResult:
        return _ToolResult(
            tool_name="interaction_checker",
            success=False,
            error="connection refused",
            error_type="MCPError",
        )


async def test_an_unreachable_mcp_server_is_named_in_the_report_not_a_generic_failure() -> None:
    drug_safety = _RealDrugSafety(
        llm=MockLLMProvider(),
        interaction_checker=_McpDownChecker(),  # type: ignore[arg-type]
    )
    evidence = AsyncMock()
    evidence.gather_evidence.return_value = _agent_result(AgentRole.EVIDENCE, "Evidence text.")

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "Unavailable: Drug Safety (an external medical data source was unavailable)" in report
    assert "could not be completed" not in report


async def test_specialist_steps_are_counted_by_outcome() -> None:
    def total(role: str, outcome: str) -> float:
        return metric_value("medagent_agent_run_total", agent_role=role, outcome=outcome)

    before = {
        "ok": total("drug_safety", "completed"),
        "failed": total("evidence", "failed"),
    }
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError("down")

    await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert total("drug_safety", "completed") - before["ok"] == 1
    assert total("evidence", "failed") - before["failed"] == 1


# --- rate limits ---------------------------------------------------------------------------


def _throttled(retry_after: float | None = None, error_type: type[Exception] = ProviderError):
    return error_type(
        "Groq 429 org_01SECRET", reason="rate_limited", retryable=True, retry_after=retry_after
    )


async def test_a_rate_limited_specialist_is_reported_as_rate_limited_not_unavailable() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = _throttled(retry_after=7.2)

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "the language model service is rate-limited; try again in about 8 seconds" in report
    assert "unavailable" not in report.split("## Note")[-1]
    assert "org_01SECRET" not in report


async def test_a_long_rate_limit_is_worded_in_minutes() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = _throttled(retry_after=487)

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "try again in about 9 minutes" in report


async def test_an_unknown_wait_says_shortly() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.return_value = _agent_result(AgentRole.DRUG_SAFETY, "ok")
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = _throttled(retry_after=None)

    report = await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert "rate-limited; try again shortly" in report


async def test_all_specialists_rate_limited_is_a_429_with_the_longest_wait() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.side_effect = _throttled(retry_after=3)
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = _throttled(retry_after=40, error_type=MCPError)

    with pytest.raises(AllAgentsFailedError) as excinfo:
        await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 40
    assert "org_01SECRET" not in str(excinfo.value)


async def test_a_mix_of_rate_limit_and_outage_is_still_a_502() -> None:
    drug_safety = AsyncMock()
    drug_safety.run_result.side_effect = _throttled(retry_after=3)
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError("down", reason="server_error")

    with pytest.raises(AllAgentsFailedError) as excinfo:
        await _assess_with(drug_safety, evidence, AsyncMock(), _BOTH)

    assert excinfo.value.status_code == 502
    assert excinfo.value.retry_after is None

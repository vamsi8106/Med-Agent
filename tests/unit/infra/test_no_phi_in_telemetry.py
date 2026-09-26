"""The telemetry rule, enforced: no patient identifier, name, medication,
condition, allergy, or question text in any span, span event, or log call.

Distinctive strings are pushed through the real agents and the real Groq
provider (with only the network faked). A positive control proves they really
did flow through the system -- they are in the report -- so an empty result
cannot be a vacuous pass.
"""

import json
from unittest.mock import AsyncMock

from opentelemetry import trace

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.models import ClinicalEvidence, DrugInteraction, Medication, PatientContext
from medagent.core.types import InteractionSeverity
from medagent.tools.base import ToolResult
from medagent.workflows.patient_assessment import run_patient_assessment
from tests.conftest import FakeClock, span_text
from tests.unit.test_groq_provider import (
    _fake_groq_response,
    _FakeSDK,
    _ok_response,
    _provider,
    _tool_call,
)

# Deliberately unlike any real word, so a match can only be a leak.
_SENSITIVE = ["zzyzx", "quillfeather", "zorbaflex", "blorptin", "glimmerfruit", "snorkelitis"]
_PATIENT_ID = "P-ZZ-LEAKCHECK-9911"


class _Checker:
    async def run(self, medications: list) -> ToolResult:
        from datetime import UTC, datetime

        return ToolResult(
            tool_name="interaction_checker",
            success=True,
            data=[
                DrugInteraction(
                    drug_a="a",
                    drug_b="b",
                    severity=InteractionSeverity.MODERATE,
                    description="generic interaction text",
                    source="test",
                    checked_at=datetime.now(UTC),
                )
            ],
        )


class _MedicalClient:
    def __init__(self) -> None:
        self.search_medical_literature = AsyncMock(
            return_value=[{"text": "Generic literature about a class of drugs."}]
        )

    async def __aenter__(self) -> "_MedicalClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


async def test_no_patient_data_reaches_any_span_or_log_call(
    spans, recorded_logs, clock: FakeClock
) -> None:
    patient = PatientContext(
        id=_PATIENT_ID,
        name="Zzyzx Quillfeather",
        age=61,
        sex="F",
        conditions=["Snorkelitis"],
        allergies=["Glimmerfruit"],
        medications=[Medication(name="Zorbaflex"), Medication(name="Blorptin")],
    )
    question = (
        "Any interaction concerns or treatment evidence for Zorbaflex with Glimmerfruit "
        "in Quillfeather with Snorkelitis?"
    )
    # Evidence: the model searches (echoing patient words into its queries, as a
    # real model would), then answers mentioning the patient.
    evidence_sdk = _FakeSDK(
        _fake_groq_response(
            [
                _tool_call("c1", "search_medical_literature", '{"query": "Zorbaflex Snorkelitis"}'),
                _tool_call("c2", "search_guidelines", '{"query": "Glimmerfruit Quillfeather"}'),
            ]
        ),
        _fake_groq_response(None, content="For Quillfeather, Zorbaflex looks reasonable."),
    )
    evidence_provider, _ = _provider(evidence_sdk)
    drug_provider, _ = _provider(_FakeSDK(_ok_response()))
    guidelines = AsyncMock()
    guidelines.run.return_value = [
        ClinicalEvidence(title="Generic guideline", summary="generic text", source="src")
    ]

    with trace.get_tracer("t").start_as_current_span("request"):
        report = await run_patient_assessment(
            TriageAgent(),
            DrugSafetyAgent(drug_provider, _Checker()),  # type: ignore[arg-type]
            EvidenceAgent(evidence_provider, _MedicalClient(), guidelines),  # type: ignore[arg-type]
            AsyncMock(),
            ReportAgent(),
            patient,
            question,
        )

    # Positive control: the sensitive data really did flow through the system.
    assert "quillfeather" in report.lower()
    assert evidence_sdk.calls == 2
    finished = spans.get_finished_spans()
    assert {"llm.chat", "react.turn", "react.tool", "node.report"} <= {s.name for s in finished}

    telemetry = " ".join(span_text(s) for s in finished)
    telemetry += " " + json.dumps(recorded_logs, default=str).lower()
    leaks = [token for token in [*_SENSITIVE, _PATIENT_ID.lower()] if token in telemetry]
    assert leaks == [], f"patient data leaked into telemetry: {leaks}"

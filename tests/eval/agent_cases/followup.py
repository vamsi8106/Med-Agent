"""Follow-up visits: the record at the last visit is saved with a visit
snapshot, then the current record, and the follow-up workflow runs through
its approval step. The report must show what changed (deterministic, from the
snapshot diff) and the answer must take the change into account."""

from datetime import UTC, datetime

from medagent.core.models import LabResult, Medication, PatientContext, Visit
from medagent.core.types import AgentRole, InteractionSeverity
from tests.eval.agent_cases.base import AgentCase

_LAST_VISIT = datetime(2026, 9, 15, tzinfo=UTC)


def _egfr(value: float, when: datetime) -> LabResult:
    return LabResult(
        test_name="eGFR",
        value=value,
        unit="mL/min/1.73m2",
        reference_low=90,
        is_abnormal=True,
        collected_at=when,
    )


_FU1 = PatientContext(
    id="P-EVAL-901", name="Patient Nu Two", age=61, sex="M", conditions=["hypertension"]
)
_FU2 = PatientContext(
    id="P-EVAL-902",
    name="Patient Xi Two",
    age=68,
    sex="F",
    conditions=["type 2 diabetes", "chronic kidney disease"],
)
_FU3 = PatientContext(
    id="P-EVAL-903", name="Patient Omicron Two", age=74, sex="M", conditions=["atrial fibrillation"]
)

CASES: list[AgentCase] = [
    AgentCase(
        id="fu-01",
        category="followup",
        workflow="followup",
        previous=_FU1.model_copy(
            update={
                "medications": [Medication(name="Lisinopril", dose="10 mg", frequency="daily")],
                "visits": [
                    Visit(
                        visit_date=_LAST_VISIT,
                        chief_complaint="dry cough",
                        assessment="Dry cough since starting lisinopril.",
                    )
                ],
            }
        ),
        patient=_FU1.model_copy(
            update={"medications": [Medication(name="Losartan", dose="50 mg", frequency="daily")]}
        ),
        message="Follow-up visit: what do you recommend now for his blood pressure?",
        source="NICE NG136: offer an ARB if an ACE inhibitor is not tolerated",
        expected_roles=frozenset({AgentRole.EVIDENCE}),
        must_show_changes=("losartan", "lisinopril"),
        must_mention=(("losartan", "arb", "angiotensin"),),
    ),
    AgentCase(
        id="fu-02",
        category="followup",
        workflow="followup",
        previous=_FU2.model_copy(
            update={
                "medications": [
                    Medication(name="Metformin", dose="1000 mg", frequency="twice daily")
                ],
                "lab_results": [_egfr(58, datetime(2026, 6, 1, tzinfo=UTC))],
            }
        ),
        patient=_FU2.model_copy(
            update={
                "medications": [
                    Medication(name="Metformin", dose="1000 mg", frequency="twice daily")
                ],
                "lab_results": [_egfr(38, datetime(2026, 11, 1, tzinfo=UTC))],
            }
        ),
        message="Follow-up: what does the guideline recommend for her metformin now?",
        source="FDA metformin label: reassess benefit at eGFR 30-45, do not start below 45",
        expected_roles=frozenset({AgentRole.EVIDENCE}),
        must_show_changes=("egfr",),
        must_mention=(("metformin",), ("egfr", "kidney", "renal")),
    ),
    AgentCase(
        id="fu-03",
        category="followup",
        workflow="followup",
        previous=_FU3.model_copy(
            update={"medications": [Medication(name="Warfarin", dose="5 mg", frequency="daily")]}
        ),
        patient=_FU3.model_copy(
            update={
                "medications": [
                    Medication(name="Warfarin", dose="5 mg", frequency="daily"),
                    Medication(name="Aspirin", dose="75 mg", frequency="daily"),
                ]
            }
        ),
        message="Check interactions since the last visit.",
        source="FDA warfarin label (bleeding risk with antiplatelet agents)",
        expected_roles=frozenset({AgentRole.DRUG_SAFETY}),
        must_show_changes=("aspirin",),
        drug_pair=("Warfarin", "Aspirin"),
        min_severity=InteractionSeverity.MODERATE,
        clinical_severity=InteractionSeverity.MAJOR,
    ),
]

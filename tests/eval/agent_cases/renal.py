"""Kidney disease: answers must adapt to a recorded CKD and, for guideline
questions, come from the ingested KDIGO guideline (search_guidelines)."""

from datetime import UTC, datetime

from medagent.core.models import LabResult, Medication, PatientContext
from medagent.core.types import AgentRole
from tests.eval.agent_cases.base import AgentCase

_EVIDENCE = frozenset({AgentRole.EVIDENCE})
_GUIDELINES = frozenset({"search_guidelines"})


def _egfr(value: float) -> LabResult:
    return LabResult(
        test_name="eGFR",
        value=value,
        unit="mL/min/1.73m2",
        reference_low=90,
        is_abnormal=True,
        collected_at=datetime(2026, 10, 1, tzinfo=UTC),
    )


CASES: list[AgentCase] = [
    AgentCase(
        id="rn-01",
        category="renal",
        patient=PatientContext(
            id="P-EVAL-401",
            name="Patient Xi",
            age=64,
            sex="M",
            conditions=["chronic kidney disease"],
        ),
        message="What's the first-line treatment for type 2 diabetes?",
        source="FDA metformin label (renal impairment); KDIGO 2024",
        expected_roles=_EVIDENCE,
        must_mention=(("renal", "kidney", "ckd", "egfr"),),
        notes="Moved from the old evidence eval: the usual answer needs renal caveats.",
    ),
    AgentCase(
        id="rn-02",
        category="renal",
        patient=PatientContext(
            id="P-EVAL-402",
            name="Patient Omicron",
            age=69,
            sex="F",
            conditions=["type 2 diabetes", "chronic kidney disease stage 4"],
            medications=[Medication(name="Metformin", dose="1000 mg", frequency="twice daily")],
            lab_results=[_egfr(25)],
        ),
        message="What does the guideline recommend for her metformin at this eGFR?",
        source="FDA metformin label: contraindicated at eGFR below 30",
        expected_roles=_EVIDENCE,
        expected_tools=_GUIDELINES,
        must_mention=(
            ("metformin",),
            ("egfr", "renal", "kidney"),
            ("stop", "discontinu", "contraindicat", "avoid", "not recommended", "do not"),
        ),
    ),
    AgentCase(
        id="rn-03",
        category="renal",
        patient=PatientContext(
            id="P-EVAL-403",
            name="Patient Pi",
            age=73,
            sex="F",
            conditions=["chronic kidney disease stage 3b", "knee osteoarthritis"],
            lab_results=[_egfr(38)],
        ),
        message="Is ibuprofen a reasonable treatment for her knee pain?",
        source="KDIGO 2024 CKD guideline: avoid NSAIDs (nephrotoxicity, AKI risk)",
        expected_roles=_EVIDENCE,
        must_mention=(
            ("nsaid", "ibuprofen"),
            ("kidney", "renal", "ckd", "egfr"),
            ("avoid", "caution", "risk", "not recommended"),
        ),
    ),
    AgentCase(
        id="rn-04",
        category="renal",
        patient=PatientContext(
            id="P-EVAL-404",
            name="Patient Rho",
            age=61,
            sex="M",
            conditions=["type 2 diabetes", "chronic kidney disease stage 3a"],
            lab_results=[_egfr(52)],
        ),
        message="What treatment do the guidelines recommend to slow his kidney disease?",
        source="KDIGO 2024 CKD guideline: SGLT2 inhibitor for CKD with type 2 diabetes",
        expected_roles=_EVIDENCE,
        expected_tools=_GUIDELINES,
        must_mention=(
            ("sglt2", "sglt-2", "empagliflozin", "dapagliflozin", "canagliflozin", "gliflozin"),
        ),
    ),
    AgentCase(
        id="rn-05",
        category="renal",
        patient=PatientContext(
            id="P-EVAL-405",
            name="Patient Sigma",
            age=57,
            sex="F",
            conditions=["hypertension", "chronic kidney disease with albuminuria"],
            lab_results=[_egfr(48)],
        ),
        message="Which antihypertensive treatment do the guidelines recommend for her?",
        source="KDIGO 2024 CKD guideline: RAS inhibitor (ACE inhibitor or ARB) with albuminuria",
        expected_roles=_EVIDENCE,
        expected_tools=_GUIDELINES,
        must_mention=(("ace inhibitor", "acei", "arb", "angiotensin", "ras inhibitor", "raas"),),
    ),
]

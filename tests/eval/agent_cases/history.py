"""Follow-up context: the answer must use what earlier visits recorded."""

from datetime import UTC, datetime

from medagent.core.models import Medication, PatientContext, Visit
from medagent.core.types import AgentRole
from tests.eval.agent_cases.base import AgentCase

_EVIDENCE = frozenset({AgentRole.EVIDENCE})
_LAST_VISIT = datetime(2026, 9, 15, tzinfo=UTC)

CASES: list[AgentCase] = [
    AgentCase(
        id="hx-01",
        category="history",
        patient=PatientContext(
            id="P-EVAL-601",
            name="Patient Chi",
            age=45,
            sex="F",
            visits=[
                Visit(
                    visit_date=_LAST_VISIT,
                    chief_complaint="fatigue",
                    assessment="Suspected anemia, ordered CBC and iron studies.",
                )
            ],
        ),
        message="Any updates on her workup?",
        source="Prior visit record",
        expected_roles=_EVIDENCE,
        must_mention=(("anemia", "anaemia", "iron", "cbc", "fatigue"),),
        notes="Moved from the old evidence eval.",
    ),
    AgentCase(
        id="hx-02",
        category="history",
        patient=PatientContext(
            id="P-EVAL-602",
            name="Patient Psi",
            age=58,
            sex="M",
            conditions=["hypertension"],
            medications=[Medication(name="Lisinopril", dose="10 mg", frequency="daily")],
            visits=[
                Visit(
                    visit_date=_LAST_VISIT,
                    chief_complaint="raised blood pressure",
                    assessment="Started lisinopril 10 mg daily for hypertension.",
                )
            ],
        ),
        message="He has had a dry cough since the last visit. What treatment do you recommend?",
        source="FDA lisinopril label (cough); NICE NG136: offer an ARB if an ACE inhibitor "
        "is not tolerated",
        expected_roles=_EVIDENCE,
        must_mention=(
            ("lisinopril", "ace inhibitor", "acei"),
            ("arb", "angiotensin receptor", "angiotensin ii receptor", "losartan", "candesartan"),
        ),
    ),
    AgentCase(
        id="hx-03",
        category="history",
        patient=PatientContext(
            id="P-EVAL-603",
            name="Patient Omega",
            age=55,
            sex="F",
            conditions=["type 2 diabetes"],
            medications=[Medication(name="Metformin", dose="1000 mg", frequency="twice daily")],
            visits=[
                Visit(
                    visit_date=_LAST_VISIT,
                    chief_complaint="diabetes review",
                    assessment="HbA1c 9.1%. Metformin increased to 1000 mg twice daily.",
                )
            ],
        ),
        message="What would you recommend next for her diabetes?",
        source="Prior visit record (HbA1c above target on metformin)",
        expected_roles=_EVIDENCE,
        must_mention=(("hba1c", "a1c", "glycemic", "glycaemic", "glucose"),),
    ),
    AgentCase(
        id="hx-04",
        category="history",
        patient=PatientContext(
            id="P-EVAL-604",
            name="Patient Alpha Two",
            age=76,
            sex="M",
            conditions=["atrial fibrillation"],
            medications=[Medication(name="Warfarin", dose="5 mg", frequency="daily")],
            visits=[
                Visit(
                    visit_date=_LAST_VISIT,
                    chief_complaint="routine INR check",
                    assessment="INR 4.2, above range. Warfarin dose held for one day.",
                )
            ],
        ),
        message="What does the evidence recommend for his anticoagulation follow-up?",
        source="Prior visit record (supratherapeutic INR)",
        expected_roles=_EVIDENCE,
        must_mention=(("inr",), ("warfarin",)),
    ),
]

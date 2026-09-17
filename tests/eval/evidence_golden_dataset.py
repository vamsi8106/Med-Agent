"""Golden evidence-lookup cases that specifically require the LLM to have
used patient-record context (conditions, allergies, visit history), not
just answered the question generically.

Ground truth here is "did the response reference at least one context-
specific term", not exact wording -- proving the LLM actually incorporated
the patient's record, which the existing unit tests only check at the
prompt-construction level (they never call a real LLM).
"""

from datetime import UTC, datetime
from typing import NamedTuple

from medagent.core.models import PatientContext, Visit


class GoldenEvidenceCase(NamedTuple):
    patient: PatientContext
    message: str
    must_reference: list[str]
    notes: str


CASES: list[GoldenEvidenceCase] = [
    GoldenEvidenceCase(
        patient=PatientContext(
            id="P-EVAL-101",
            name="Patient Alpha",
            age=64,
            sex="M",
            conditions=["chronic kidney disease"],
        ),
        message="What's the first-line treatment for type 2 diabetes?",
        must_reference=["renal", "kidney", "contraindicated", "ckd"],
        notes=(
            "Metformin (the usual first-line answer) genuinely needs renal "
            "caveats in a CKD patient -- a generic answer that ignores the "
            "recorded condition would miss this."
        ),
    ),
    GoldenEvidenceCase(
        patient=PatientContext(
            id="P-EVAL-102",
            name="Patient Beta",
            age=45,
            sex="F",
            visits=[
                Visit(
                    visit_date=datetime.now(UTC),
                    chief_complaint="fatigue",
                    assessment="Suspected anemia, ordered CBC and iron studies.",
                )
            ],
        ),
        message="Any updates on her workup?",
        must_reference=["anemia", "iron", "cbc", "fatigue"],
        notes=(
            "A generic 'any updates' answer with no visit history to draw on "
            "would be vague; awareness of the prior visit's suspected "
            "diagnosis should surface specific, relevant terms."
        ),
    ),
    GoldenEvidenceCase(
        patient=PatientContext(
            id="P-EVAL-103",
            name="Patient Gamma",
            age=52,
            sex="F",
            allergies=["Penicillin"],
        ),
        message="What antibiotic would you recommend for a routine sinus infection?",
        must_reference=["penicillin", "allerg", "avoid", "alternative"],
        notes="A recorded penicillin allergy should shape which antibiotic class is discussed.",
    ),
]

"""Allergy conflicts: a drug the patient is recorded as allergic to must be
flagged (`⚠️ ALLERGY CONFLICT`) in every run. Drug Safety's check is
deterministic (allergy name vs drug name), so the named cases are gated at
100 %. Cross-reactivity (amoxicillin is a penicillin) is not implemented, so
that case is a known gap: reported, not gated.
"""

from medagent.core.models import Medication, PatientContext
from medagent.core.types import AgentRole
from tests.eval.agent_cases.base import AgentCase


def _named_allergy_case(
    case_id: str, name: str, age: int, sex: str, drugs: tuple[str, str], allergy: str
) -> AgentCase:
    return AgentCase(
        id=case_id,
        category="allergy",
        patient=PatientContext(
            id=case_id.replace("al-", "P-EVAL-3"),
            name=name,
            age=age,
            sex=sex,
            allergies=[allergy],
            medications=[Medication(name=drugs[0]), Medication(name=drugs[1])],
        ),
        message=f"Check interactions for {drugs[0]} and {drugs[1]}",
        source="Recorded allergy matches a listed drug by name",
        expected_roles=frozenset({AgentRole.DRUG_SAFETY}),
        drug_pair=drugs,
        expect_allergy_flag=True,
    )


CASES: list[AgentCase] = [
    _named_allergy_case("al-01", "Patient Iota", 71, "F", ("Warfarin", "Aspirin"), "Aspirin"),
    _named_allergy_case(
        "al-02", "Patient Kappa", 58, "M", ("Lisinopril", "Ibuprofen"), "Ibuprofen"
    ),
    _named_allergy_case("al-03", "Patient Lambda", 47, "F", ("Sertraline", "Codeine"), "Codeine"),
    AgentCase(
        id="al-04",
        category="allergy",
        patient=PatientContext(
            id="P-EVAL-304",
            name="Patient Mu",
            age=52,
            sex="M",
            allergies=["Penicillin"],
            medications=[Medication(name="Metformin"), Medication(name="Amoxicillin")],
        ),
        message="Check interactions for Metformin and Amoxicillin",
        source="Amoxicillin is an aminopenicillin (FDA amoxicillin label: contraindicated "
        "in penicillin hypersensitivity)",
        expected_roles=frozenset({AgentRole.DRUG_SAFETY}),
        drug_pair=("Metformin", "Amoxicillin"),
        expect_allergy_flag=True,
        known_gap="Allergy check matches names only; no drug-class cross-reactivity.",
    ),
    AgentCase(
        id="al-05",
        category="allergy",
        patient=PatientContext(
            id="P-EVAL-305",
            name="Patient Nu",
            age=52,
            sex="F",
            allergies=["Penicillin"],
        ),
        message="What antibiotic would you recommend for a routine sinus infection?",
        source="Recorded penicillin allergy should shape the antibiotic discussed",
        expected_roles=frozenset({AgentRole.EVIDENCE}),
        must_mention=(("penicillin", "allerg"),),
        notes="Quality case (LLM answer); the flag is only gated for Drug Safety's check.",
    ),
]

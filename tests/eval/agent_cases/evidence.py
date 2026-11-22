"""Evidence questions whose right answer depends on the patient's record."""

from medagent.core.models import Medication, PatientContext
from medagent.core.types import AgentRole
from tests.eval.agent_cases.base import AgentCase

_EVIDENCE = frozenset({AgentRole.EVIDENCE})

CASES: list[AgentCase] = [
    AgentCase(
        id="ev-01",
        category="evidence",
        patient=PatientContext(
            id="P-EVAL-501",
            name="Patient Tau",
            age=62,
            sex="F",
            conditions=["hypertension"],
        ),
        message="What first-line treatment does the guideline recommend for her hypertension?",
        source="NICE NG136: age 55 or over (no type 2 diabetes) -> calcium-channel blocker",
        expected_roles=_EVIDENCE,
        expected_tools=frozenset({"search_guidelines"}),
        must_mention=(("calcium-channel", "calcium channel", "ccb", "amlodipine"),),
    ),
    AgentCase(
        id="ev-02",
        category="evidence",
        patient=PatientContext(
            id="P-EVAL-502",
            name="Patient Upsilon",
            age=82,
            sex="F",
            conditions=["insomnia", "osteoporosis", "hypertension"],
            medications=[
                Medication(name="Diazepam", dose="5 mg", frequency="nightly"),
                Medication(name="Amlodipine", dose="5 mg"),
                Medication(name="Alendronate", dose="70 mg", frequency="weekly"),
            ],
        ),
        message="Any evidence-based recommendation about her diazepam?",
        source="AGS Beers Criteria: avoid benzodiazepines in older adults (falls, cognition)",
        expected_roles=_EVIDENCE,
        must_mention=(
            ("fall", "deprescrib", "taper", "beers", "older adult", "elderly", "sedation"),
        ),
    ),
    AgentCase(
        id="ev-03",
        category="evidence",
        patient=PatientContext(
            id="P-EVAL-503",
            name="Patient Phi",
            age=63,
            sex="M",
            conditions=["type 2 diabetes", "coronary artery disease"],
            medications=[Medication(name="Metformin", dose="1000 mg", frequency="twice daily")],
        ),
        message="What treatment do guidelines recommend for his diabetes given his heart disease?",
        source="Cardiovascular-outcome trials: GLP-1 receptor agonist or SGLT2 inhibitor in "
        "type 2 diabetes with established ASCVD",
        expected_roles=_EVIDENCE,
        must_mention=(
            (
                "glp-1",
                "glp1",
                "sglt2",
                "sglt-2",
                "semaglutide",
                "liraglutide",
                "dulaglutide",
                "empagliflozin",
                "dapagliflozin",
            ),
        ),
    ),
]

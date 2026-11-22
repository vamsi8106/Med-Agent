"""Drug-pair checks: Drug Safety must report the pair, never below the gate
floor (`min_severity`), and is scored against the clinically expected
severity (`clinical_severity`).

The floor is what the interaction source must at least say for the check to
count as done (MODERATE: the pair was analysed). `clinical_severity` is the
documented severity; where the live source disagrees, the report counts the
case as under- or over-graded. Two controls with no known interaction measure
over-grading.
"""

from medagent.core.models import Medication, PatientContext
from medagent.core.types import AgentRole, InteractionSeverity
from tests.eval.agent_cases.base import AgentCase

_DRUG_SAFETY = frozenset({AgentRole.DRUG_SAFETY})


def _pair_case(
    case_id: str,
    name: str,
    age: int,
    sex: str,
    pair: tuple[str, str],
    clinical: InteractionSeverity | None,
    source: str,
    notes: str,
    min_severity: InteractionSeverity | None = InteractionSeverity.MODERATE,
) -> AgentCase:
    return AgentCase(
        id=case_id,
        category="interactions",
        patient=PatientContext(
            id=case_id.replace("ix-", "P-EVAL-2"),
            name=name,
            age=age,
            sex=sex,
            medications=[Medication(name=pair[0]), Medication(name=pair[1])],
        ),
        message=f"Check interactions for {pair[0]} and {pair[1]}",
        source=source,
        expected_roles=_DRUG_SAFETY,
        must_mention=((pair[0].lower(),), (pair[1].lower(),)),
        drug_pair=pair,
        min_severity=min_severity,
        clinical_severity=clinical,
        notes=notes,
    )


CASES: list[AgentCase] = [
    _pair_case(
        "ix-01",
        "Patient Alpha",
        70,
        "M",
        ("Warfarin", "Aspirin"),
        InteractionSeverity.MAJOR,
        "FDA warfarin label (bleeding risk with antiplatelet agents)",
        "Additive anticoagulant/antiplatelet effect: major bleeding risk.",
    ),
    _pair_case(
        "ix-02",
        "Patient Beta",
        68,
        "F",
        ("Metformin", "Glimepiride"),
        None,
        "Commonly co-prescribed; additive glucose lowering",
        "Severity left unset: references differ (minor to moderate hypoglycaemia risk).",
    ),
    _pair_case(
        "ix-03",
        "Patient Gamma",
        62,
        "M",
        ("Lisinopril", "Spironolactone"),
        InteractionSeverity.MAJOR,
        "FDA spironolactone label (hyperkalaemia with ACE inhibitors)",
        "ACE inhibitor + potassium-sparing diuretic: hyperkalaemia.",
    ),
    _pair_case(
        "ix-04",
        "Patient Delta",
        59,
        "F",
        ("Simvastatin", "Clarithromycin"),
        InteractionSeverity.CONTRAINDICATED,
        "FDA simvastatin label: strong CYP3A4 inhibitors incl. clarithromycin contraindicated",
        "Myopathy/rhabdomyolysis risk.",
    ),
    _pair_case(
        "ix-05",
        "Patient Epsilon",
        41,
        "F",
        ("Sertraline", "Phenelzine"),
        InteractionSeverity.CONTRAINDICATED,
        "FDA sertraline label: use with MAOIs contraindicated",
        "Serotonin syndrome.",
    ),
    _pair_case(
        "ix-06",
        "Patient Zeta",
        66,
        "M",
        ("Sildenafil", "Nitroglycerin"),
        InteractionSeverity.CONTRAINDICATED,
        "FDA sildenafil label: contraindicated with nitrates",
        "Severe hypotension.",
    ),
    _pair_case(
        "ix-07",
        "Patient Eta",
        35,
        "F",
        ("Acetaminophen", "Amoxicillin"),
        InteractionSeverity.NONE,
        "No interaction listed in either FDA label",
        "Control: a pair with no known interaction (measures over-grading).",
        min_severity=None,
    ),
    _pair_case(
        "ix-08",
        "Patient Theta",
        29,
        "M",
        ("Loratadine", "Cholecalciferol"),
        InteractionSeverity.NONE,
        "No interaction listed in either FDA label",
        "Control: a pair with no known interaction (measures over-grading).",
        min_severity=None,
    ),
]

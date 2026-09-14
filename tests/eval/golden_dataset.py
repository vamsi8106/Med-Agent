"""Golden drug-interaction cases with well-documented minimum severities.

Ground truth here is a *minimum acceptable severity*, not exact text: real
MCP responses vary in wording, but a textbook-dangerous interaction must
never be reported weaker than its established clinical severity.
"""

from typing import NamedTuple

from medagent.core.models import Medication, PatientContext
from medagent.core.types import InteractionSeverity


class GoldenCase(NamedTuple):
    patient: PatientContext
    drugs: list[str]
    min_expected_severity: InteractionSeverity
    notes: str


CASES: list[GoldenCase] = [
    GoldenCase(
        patient=PatientContext(
            id="P-EVAL-001",
            name="Patient Alpha",
            age=70,
            sex="M",
            medications=[Medication(name="Warfarin"), Medication(name="Aspirin")],
        ),
        drugs=["Warfarin", "Aspirin"],
        min_expected_severity=InteractionSeverity.MAJOR,
        notes="Combined anticoagulant/antiplatelet effect: major bleeding risk.",
    ),
    GoldenCase(
        patient=PatientContext(
            id="P-EVAL-002",
            name="Patient Beta",
            age=68,
            sex="F",
            conditions=["type 2 diabetes"],
            medications=[Medication(name="Metformin"), Medication(name="Glimepiride")],
        ),
        drugs=["Metformin", "Glimepiride"],
        min_expected_severity=InteractionSeverity.MODERATE,
        notes="Additive glucose-lowering effect: moderate hypoglycemia risk.",
    ),
    GoldenCase(
        patient=PatientContext(
            id="P-EVAL-003",
            name="Patient Gamma",
            age=62,
            sex="M",
            conditions=["hypertension"],
            medications=[Medication(name="Lisinopril"), Medication(name="Spironolactone")],
        ),
        drugs=["Lisinopril", "Spironolactone"],
        min_expected_severity=InteractionSeverity.MODERATE,
        notes="ACE inhibitor + potassium-sparing diuretic: moderate hyperkalemia risk.",
    ),
    GoldenCase(
        patient=PatientContext(
            id="P-EVAL-004",
            name="Patient Delta",
            age=55,
            sex="F",
            medications=[Medication(name="Simvastatin"), Medication(name="Clarithromycin")],
        ),
        drugs=["Simvastatin", "Clarithromycin"],
        min_expected_severity=InteractionSeverity.MAJOR,
        notes="CYP3A4 inhibition raises statin levels: major rhabdomyolysis risk.",
    ),
]

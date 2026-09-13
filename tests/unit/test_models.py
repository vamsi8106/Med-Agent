from datetime import UTC, datetime

from medagent.core.models import DrugInteraction, Medication, PatientContext
from medagent.core.types import InteractionSeverity


def test_patient_context_defaults() -> None:
    patient = PatientContext(id="P-TEST-001", name="Patient Alpha", age=54, sex="F")
    assert patient.conditions == []
    assert patient.medications == []


def test_medication_optional_fields() -> None:
    med = Medication(name="Metformin")
    assert med.status == "active"
    assert med.dose is None


def test_drug_interaction_severity_enum() -> None:
    interaction = DrugInteraction(
        drug_a="Metformin",
        drug_b="Glimepiride",
        severity=InteractionSeverity.MODERATE,
        description="Increased hypoglycemia risk",
        checked_at=datetime.now(UTC),
    )
    assert interaction.severity == InteractionSeverity.MODERATE

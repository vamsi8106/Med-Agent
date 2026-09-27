"""What changed in a patient's record since the last visit.

Each visit stores a snapshot of the record (see PatientStore.save_visit); a
follow-up compares the current record with the newest snapshot. Deterministic
on purpose: which medications started or stopped and which labs have newer
values is a fact to compute, not something to ask the model.
"""

from datetime import datetime
from typing import Any

from medagent.core.models import (
    LabChange,
    LabResult,
    Medication,
    MedicationChange,
    PatientContext,
    RecordChanges,
)


def medication_key(name: str) -> str:
    return name.strip().lower()


def regimen(med: Medication | dict[str, Any]) -> str:
    """Dose, frequency and route as one comparable string."""
    get = med.get if isinstance(med, dict) else lambda field: getattr(med, field)
    parts = [get(field) for field in ("dose", "frequency", "route")]
    return " ".join(str(part) for part in parts if part) or "unspecified"


def record_snapshot(context: PatientContext) -> dict[str, Any]:
    """The record as it stands: active medications and the latest lab per test."""
    return {
        "conditions": list(context.conditions),
        "allergies": list(context.allergies),
        "medications": [
            {"name": m.name, "dose": m.dose, "frequency": m.frequency, "route": m.route}
            for m in context.medications
        ],
        "labs": [
            {
                "test_name": lab.test_name,
                "value": lab.value,
                "unit": lab.unit,
                "collected_at": lab.collected_at.isoformat(),
            }
            for lab in context.lab_results
        ],
    }


def _added(before: list[str], after: list[str]) -> list[str]:
    seen = {item.strip().lower() for item in before}
    return [item for item in after if item.strip().lower() not in seen]


def _lab_changes(snapshot_labs: list[dict[str, Any]], current: list[LabResult]) -> list[LabChange]:
    previous = {lab["test_name"].strip().lower(): lab for lab in snapshot_labs}
    changes = []
    for lab in current:
        prior = previous.get(lab.test_name.strip().lower())
        if prior is not None and datetime.fromisoformat(prior["collected_at"]) >= lab.collected_at:
            continue
        changes.append(
            LabChange(
                test_name=lab.test_name,
                unit=lab.unit,
                previous_value=prior["value"] if prior is not None else None,
                current_value=lab.value,
                reference_low=lab.reference_low,
                reference_high=lab.reference_high,
                collected_at=lab.collected_at,
            )
        )
    return changes


def compute_changes(
    snapshot: dict[str, Any], current: PatientContext, since: datetime
) -> RecordChanges:
    before_meds = {medication_key(m["name"]): m for m in snapshot.get("medications", [])}
    after_meds = {medication_key(m.name): m for m in current.medications}

    changed = [
        MedicationChange(name=med.name, before=regimen(before_meds[key]), after=regimen(med))
        for key, med in after_meds.items()
        if key in before_meds and regimen(before_meds[key]) != regimen(med)
    ]
    return RecordChanges(
        since=since,
        medications_started=[m.name for k, m in after_meds.items() if k not in before_meds],
        medications_stopped=[m["name"] for k, m in before_meds.items() if k not in after_meds],
        medications_changed=changed,
        conditions_added=_added(snapshot.get("conditions", []), current.conditions),
        conditions_removed=_added(current.conditions, snapshot.get("conditions", [])),
        allergies_added=_added(snapshot.get("allergies", []), current.allergies),
        allergies_removed=_added(current.allergies, snapshot.get("allergies", [])),
        lab_changes=_lab_changes(snapshot.get("labs", []), current.lab_results),
    )

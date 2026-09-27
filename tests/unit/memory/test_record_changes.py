from datetime import UTC, datetime

from medagent.core.models import LabResult, Medication, PatientContext
from medagent.memory.record_changes import compute_changes, record_snapshot

_SINCE = datetime(2026, 1, 15, tzinfo=UTC)


def _patient(**overrides: object) -> PatientContext:
    fields: dict[str, object] = {
        "id": "P-TEST-001",
        "name": "Patient Alpha",
        "age": 68,
        "sex": "F",
        "conditions": ["Type 2 diabetes"],
        "allergies": ["Sulfa"],
        "medications": [Medication(name="Metformin", dose="500 mg", frequency="twice daily")],
        "lab_results": [
            LabResult(
                test_name="HbA1c",
                value=8.1,
                unit="%",
                collected_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
    }
    fields.update(overrides)
    return PatientContext(**fields)  # type: ignore[arg-type]


def test_identical_record_has_no_changes() -> None:
    patient = _patient()

    changes = compute_changes(record_snapshot(patient), patient, _SINCE)

    assert changes.is_empty
    assert changes.since == _SINCE


def test_medication_started_stopped_and_changed() -> None:
    before = _patient(
        medications=[Medication(name="Metformin", dose="500 mg"), Medication(name="Glimepiride")]
    )
    after = _patient(
        medications=[Medication(name="metformin ", dose="1000 mg"), Medication(name="Lisinopril")]
    )

    changes = compute_changes(record_snapshot(before), after, _SINCE)

    assert changes.medications_started == ["Lisinopril"]
    assert changes.medications_stopped == ["Glimepiride"]
    assert [(c.before, c.after) for c in changes.medications_changed] == [("500 mg", "1000 mg")]


def test_condition_and_allergy_matching_ignores_case() -> None:
    before = _patient(conditions=["Type 2 diabetes"], allergies=["Sulfa"])
    after = _patient(conditions=["type 2 diabetes", "CKD"], allergies=[])

    changes = compute_changes(record_snapshot(before), after, _SINCE)

    assert changes.conditions_added == ["CKD"]
    assert changes.conditions_removed == []
    assert changes.allergies_removed == ["Sulfa"]


def test_newer_lab_result_and_new_test_are_reported() -> None:
    after = _patient(
        lab_results=[
            LabResult(
                test_name="HbA1c",
                value=9.8,
                unit="%",
                collected_at=datetime(2026, 6, 1, tzinfo=UTC),
            ),
            LabResult(
                test_name="eGFR",
                value=55.0,
                unit="mL/min/1.73m2",
                collected_at=datetime(2026, 6, 1, tzinfo=UTC),
            ),
        ]
    )

    changes = compute_changes(record_snapshot(_patient()), after, _SINCE)

    assert [(c.test_name, c.previous_value, c.current_value) for c in changes.lab_changes] == [
        ("HbA1c", 8.1, 9.8),
        ("eGFR", None, 55.0),
    ]


def test_same_lab_result_is_not_a_change() -> None:
    patient = _patient()
    snapshot = record_snapshot(patient)

    assert compute_changes(snapshot, patient, _SINCE).lab_changes == []


def test_empty_snapshot_treats_everything_as_new() -> None:
    changes = compute_changes({}, _patient(), _SINCE)

    assert changes.medications_started == ["Metformin"]
    assert changes.conditions_added == ["Type 2 diabetes"]

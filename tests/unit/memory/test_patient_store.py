from datetime import UTC, datetime

import pytest

from medagent.core.exceptions import MemoryError as MedAgentMemoryError
from medagent.core.exceptions import PatientOwnershipError
from medagent.core.models import LabResult, Medication, PatientContext
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore


def _make_store(pg_dsn: str) -> PatientStore:
    return PatientStore(PersistentStore(pg_dsn))


async def test_get_missing_patient_returns_none(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    assert await store.get_patient("P-TEST-999") is None


async def test_save_then_get_round_trips_full_record(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-001",
        name="Patient Alpha",
        age=58,
        sex="F",
        doctor_id="DR-TEST-001",
        conditions=["type 2 diabetes"],
        allergies=["penicillin"],
        medications=[Medication(name="Metformin", dose="500mg")],
        lab_results=[
            LabResult(
                test_name="HbA1c",
                value=7.8,
                unit="%",
                reference_low=4.0,
                reference_high=5.7,
                is_abnormal=True,
                collected_at=datetime.now(UTC),
            )
        ],
    )

    await store.save_patient(context)
    loaded = await store.get_patient("P-TEST-001")

    assert loaded is not None
    assert loaded.name == "Patient Alpha"
    assert loaded.doctor_id == "DR-TEST-001"
    assert loaded.conditions == ["type 2 diabetes"]
    assert loaded.medications[0].name == "Metformin"
    assert loaded.lab_results[0].test_name == "HbA1c"


async def test_save_twice_updates_existing_record(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-002", name="Patient Beta", age=40, sex="M", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    context.age = 41
    await store.save_patient(context)
    loaded = await store.get_patient("P-TEST-002")

    assert loaded is not None
    assert loaded.age == 41


async def test_save_patient_without_doctor_id_raises(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(id="P-TEST-003", name="Patient Gamma", age=45, sex="F")

    with pytest.raises(MedAgentMemoryError):
        await store.save_patient(context)


async def test_get_patient_scoped_to_owning_doctor(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-004", name="Patient Delta", age=50, sex="M", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    assert (await store.get_patient("P-TEST-004", doctor_id="DR-TEST-001")) is not None
    assert (await store.get_patient("P-TEST-004", doctor_id="DR-TEST-OTHER")) is None


async def test_get_patient_unscoped_ignores_doctor_id(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-005", name="Patient Epsilon", age=33, sex="F", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    loaded = await store.get_patient("P-TEST-005")
    assert loaded is not None
    assert loaded.doctor_id == "DR-TEST-001"


async def test_save_by_another_doctor_is_refused_and_changes_nothing(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-006",
        name="Patient Zeta",
        age=29,
        sex="M",
        doctor_id="DR-TEST-001",
        medications=[Medication(name="Metformin")],
    )
    await store.save_patient(context)

    intruder = PatientContext(
        id="P-TEST-006",
        name="Overwritten",
        age=99,
        sex="F",
        doctor_id="DR-TEST-OTHER",
        medications=[Medication(name="Warfarin")],
    )
    with pytest.raises(PatientOwnershipError):
        await store.save_patient(intruder)

    loaded = await store.get_patient("P-TEST-006")
    assert loaded is not None
    assert loaded.doctor_id == "DR-TEST-001"
    assert loaded.name == "Patient Zeta"
    assert [m.name for m in loaded.medications] == ["Metformin"]


async def test_save_visit_round_trips(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-007", name="Patient Eta", age=52, sex="F", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    await store.save_visit("P-TEST-007", "DR-TEST-001", "headache", "Likely tension headache.")

    loaded = await store.get_patient("P-TEST-007")
    assert loaded is not None
    assert len(loaded.visits) == 1
    assert loaded.visits[0].chief_complaint == "headache"
    assert loaded.visits[0].assessment == "Likely tension headache."


async def test_visits_returned_newest_first(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-008", name="Patient Theta", age=48, sex="M", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    await store.save_visit("P-TEST-008", "DR-TEST-001", "first visit", "assessment 1")
    await store.save_visit("P-TEST-008", "DR-TEST-001", "second visit", "assessment 2")

    loaded = await store.get_patient("P-TEST-008")
    assert loaded is not None
    assert loaded.visits[0].chief_complaint == "second visit"
    assert loaded.visits[1].chief_complaint == "first visit"


async def test_only_most_recent_visits_returned(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-009", name="Patient Iota", age=41, sex="F", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    for i in range(7):
        await store.save_visit("P-TEST-009", "DR-TEST-001", f"visit {i}", f"assessment {i}")

    loaded = await store.get_patient("P-TEST-009")
    assert loaded is not None
    assert len(loaded.visits) == 5
    assert loaded.visits[0].chief_complaint == "visit 6"


def _alpha(**overrides: object) -> PatientContext:
    fields: dict[str, object] = {
        "id": "P-TEST-100",
        "name": "Patient Alpha",
        "age": 68,
        "sex": "F",
        "doctor_id": "DR-TEST-001",
        "conditions": ["type 2 diabetes"],
        "allergies": ["Sulfa"],
        "medications": [
            Medication(name="Metformin", dose="500 mg"),
            Medication(name="Glimepiride", dose="2 mg"),
        ],
        "lab_results": [
            LabResult(
                test_name="HbA1c",
                value=8.1,
                unit="%",
                reference_low=4.0,
                reference_high=5.7,
                collected_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        ],
    }
    fields.update(overrides)
    return PatientContext(**fields)  # type: ignore[arg-type]


async def _count(pg_dsn: str, table: str) -> int:
    store = PersistentStore(pg_dsn)
    async with store.connect() as conn:
        count: int = await conn.fetchval(f"SELECT count(*) FROM {table}")
    await store.close()
    return count


async def test_resaving_an_unchanged_record_writes_no_new_rows(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())
    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None

    await store.save_patient(loaded)
    await store.save_patient(loaded)

    assert await _count(pg_dsn, "medications") == 2
    assert await _count(pg_dsn, "lab_results") == 1


async def test_stopped_medication_keeps_its_row_with_an_end_date(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())

    await store.save_patient(_alpha(medications=[Medication(name="Metformin", dose="500 mg")]))

    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None
    assert [m.name for m in loaded.medications] == ["Metformin"]
    persistent = PersistentStore(pg_dsn)
    async with persistent.connect() as conn:
        row = await conn.fetchrow(
            "SELECT status, end_date FROM medications WHERE name = 'Glimepiride'"
        )
    await persistent.close()
    assert row["status"] == "stopped"
    assert row["end_date"] is not None


async def test_dose_change_closes_the_old_row_and_opens_a_new_one(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())
    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None

    glimepiride = next(m for m in loaded.medications if m.name == "Glimepiride")
    glimepiride.dose = "4 mg"  # the re-saved context still carries the old row's id
    await store.save_patient(loaded)

    reloaded = await store.get_patient("P-TEST-100")
    assert reloaded is not None
    assert {m.name: m.dose for m in reloaded.medications} == {
        "Metformin": "500 mg",
        "Glimepiride": "4 mg",
    }
    assert await _count(pg_dsn, "medications") == 3


async def test_labs_are_appended_and_the_latest_per_test_is_returned(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())
    newer = LabResult(
        test_name="HbA1c", value=9.8, unit="%", collected_at=datetime(2026, 6, 1, tzinfo=UTC)
    )
    await store.save_patient(_alpha(lab_results=[newer]))

    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None
    assert [(lab.test_name, lab.value) for lab in loaded.lab_results] == [("HbA1c", 9.8)]
    assert await _count(pg_dsn, "lab_results") == 2


async def test_no_changes_without_an_earlier_visit(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())

    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None
    assert loaded.changes_since_last_visit is None


async def test_changes_are_computed_against_the_last_visit(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())
    await store.save_visit("P-TEST-100", "DR-TEST-001", "initial", "# Report")

    await store.save_patient(
        _alpha(
            conditions=["type 2 diabetes", "hypertension"],
            medications=[
                Medication(name="Metformin", dose="1000 mg"),
                Medication(name="Lisinopril", dose="10 mg"),
            ],
            lab_results=[
                LabResult(
                    test_name="HbA1c",
                    value=9.8,
                    unit="%",
                    collected_at=datetime(2026, 6, 1, tzinfo=UTC),
                )
            ],
        )
    )

    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None
    changes = loaded.changes_since_last_visit
    assert changes is not None
    assert changes.medications_started == ["Lisinopril"]
    assert changes.medications_stopped == ["Glimepiride"]
    assert [(c.name, c.before, c.after) for c in changes.medications_changed] == [
        ("Metformin", "500 mg", "1000 mg")
    ]
    assert changes.conditions_added == ["hypertension"]
    assert [(c.test_name, c.previous_value, c.current_value) for c in changes.lab_changes] == [
        ("HbA1c", 8.1, 9.8)
    ]


async def test_resaving_after_a_visit_reports_no_changes(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())
    await store.save_visit("P-TEST-100", "DR-TEST-001", "initial", "# Report")

    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None
    await store.save_patient(loaded)

    reloaded = await store.get_patient("P-TEST-100")
    assert reloaded is not None
    assert reloaded.changes_since_last_visit is not None
    assert reloaded.changes_since_last_visit.is_empty


async def test_visit_without_a_snapshot_gives_no_baseline(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    await store.save_patient(_alpha())
    persistent = PersistentStore(pg_dsn)
    async with persistent.connect() as conn:
        await conn.execute(
            """INSERT INTO visits (id, patient_id, doctor_id, visit_date, created_at)
            VALUES ('V-TEST-1', 'P-TEST-100', 'DR-TEST-001', now(), now())"""
        )
    await persistent.close()

    loaded = await store.get_patient("P-TEST-100")
    assert loaded is not None
    assert loaded.changes_since_last_visit is None

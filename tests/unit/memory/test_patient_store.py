from datetime import UTC, datetime

import pytest

from medagent.core.exceptions import MemoryError as MedAgentMemoryError
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


async def test_save_twice_does_not_change_doctor_id(pg_dsn: str) -> None:
    store = _make_store(pg_dsn)
    context = PatientContext(
        id="P-TEST-006", name="Patient Zeta", age=29, sex="M", doctor_id="DR-TEST-001"
    )
    await store.save_patient(context)

    # Re-saving with a different in-memory doctor_id must not change the
    # stored owner -- the UPDATE path never touches the column.
    context.doctor_id = "DR-TEST-OTHER"
    await store.save_patient(context)

    loaded = await store.get_patient("P-TEST-006")
    assert loaded is not None
    assert loaded.doctor_id == "DR-TEST-001"


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

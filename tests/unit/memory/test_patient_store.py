from datetime import UTC, datetime
from pathlib import Path

from medagent.core.models import LabResult, Medication, PatientContext
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore


async def _make_store(tmp_path: Path) -> PatientStore:
    persistent = PersistentStore(str(tmp_path / "test.db"))
    await persistent.init_schema()
    return PatientStore(persistent)


async def test_get_missing_patient_returns_none(tmp_path: Path) -> None:
    store = await _make_store(tmp_path)
    assert await store.get_patient("P-TEST-999") is None


async def test_save_then_get_round_trips_full_record(tmp_path: Path) -> None:
    store = await _make_store(tmp_path)
    context = PatientContext(
        id="P-TEST-001",
        name="Patient Alpha",
        age=58,
        sex="F",
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
    assert loaded.conditions == ["type 2 diabetes"]
    assert loaded.medications[0].name == "Metformin"
    assert loaded.lab_results[0].test_name == "HbA1c"


async def test_save_twice_updates_existing_record(tmp_path: Path) -> None:
    store = await _make_store(tmp_path)
    context = PatientContext(id="P-TEST-002", name="Patient Beta", age=40, sex="M")
    await store.save_patient(context)

    context.age = 41
    await store.save_patient(context)
    loaded = await store.get_patient("P-TEST-002")

    assert loaded is not None
    assert loaded.age == 41

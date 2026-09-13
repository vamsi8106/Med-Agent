from medagent.core.models import PatientContext, User
from medagent.memory.audit_log import AuditLogStore
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore


def _store(pg_dsn: str) -> AuditLogStore:
    return AuditLogStore(PersistentStore(pg_dsn))


async def _seed_patient(pg_dsn: str, patient_id: str) -> None:
    patient_store = PatientStore(PersistentStore(pg_dsn))
    await patient_store.save_patient(
        PatientContext(id=patient_id, name="Patient Alpha", age=50, sex="F")
    )


async def test_record_and_read_back_entry(pg_dsn: str) -> None:
    await _seed_patient(pg_dsn, "P-TEST-900")
    store = _store(pg_dsn)
    user = User(id="U-1", username="dr.alpha", role="doctor")

    await store.record(user, "P-TEST-900", "patient_viewed")
    entries = await store.for_patient("P-TEST-900")

    assert len(entries) == 1
    assert entries[0]["action"] == "patient_viewed"
    assert entries[0]["username"] == "dr.alpha"
    assert entries[0]["details"] == {}


async def test_record_with_details(pg_dsn: str) -> None:
    await _seed_patient(pg_dsn, "P-TEST-901")
    store = _store(pg_dsn)
    user = User(id="U-2", username="dr.beta", role="doctor")

    await store.record(user, "P-TEST-901", "drug_check_run", {"new_drug": "Glimepiride"})
    entries = await store.for_patient("P-TEST-901")

    assert entries[0]["details"] == {"new_drug": "Glimepiride"}


async def test_record_without_user_is_allowed(pg_dsn: str) -> None:
    await _seed_patient(pg_dsn, "P-TEST-902")
    store = _store(pg_dsn)

    await store.record(None, "P-TEST-902", "system_event")
    entries = await store.for_patient("P-TEST-902")

    assert entries[0]["user_id"] is None
    assert entries[0]["username"] is None


async def test_entries_ordered_most_recent_first(pg_dsn: str) -> None:
    await _seed_patient(pg_dsn, "P-TEST-903")
    store = _store(pg_dsn)
    user = User(id="U-3", username="dr.gamma", role="doctor")

    await store.record(user, "P-TEST-903", "patient_viewed")
    await store.record(user, "P-TEST-903", "assessment_run")
    entries = await store.for_patient("P-TEST-903")

    assert [e["action"] for e in entries] == ["assessment_run", "patient_viewed"]

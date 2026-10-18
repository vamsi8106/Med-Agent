from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from medagent.core.config import Settings
from medagent.core.models import LabResult, Medication, PatientContext, User
from medagent.memory.audit_log import AuditLogStore
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.memory.retention import purge_expired

_NOW = datetime(2036, 1, 1, tzinfo=UTC)
_USER = User(id="U-TEST-1", username="dr.alpha", role="doctor")


async def _seed(store: PersistentStore, patient_id: str) -> None:
    await PatientStore(store).save_patient(
        PatientContext(
            id=patient_id,
            name="Patient Alpha",
            age=60,
            sex="F",
            doctor_id="DR-TEST-001",
            medications=[Medication(name="Metformin")],
            lab_results=[
                LabResult(
                    test_name="HbA1c",
                    value=8.1,
                    unit="%",
                    collected_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            ],
        )
    )
    await PatientStore(store).save_visit(patient_id, "DR-TEST-001", "check", "# Report")


async def _backdate(store: PersistentStore, patient_id: str, when: datetime) -> None:
    async with store.connect() as conn:
        await conn.execute("UPDATE patients SET updated_at = $1 WHERE id = $2", when, patient_id)
        await conn.execute(
            "UPDATE visits SET visit_date = $1 WHERE patient_id = $2", when, patient_id
        )


async def _count(store: PersistentStore, table: str, patient_id: str) -> int:
    async with store.connect() as conn:
        count: int = await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE patient_id = $1", patient_id
        )
    return count


async def test_dry_run_counts_but_deletes_nothing(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    await _seed(store, "P-TEST-OLD")
    await _backdate(store, "P-TEST-OLD", _NOW - timedelta(days=4000))

    result = await purge_expired(
        store, retention_days=3650, audit_retention_days=None, execute=False, now=_NOW
    )

    assert result.patients == 1
    assert result.audit_entries is None
    assert await _count(store, "visits", "P-TEST-OLD") == 1
    await store.close()


async def test_execute_deletes_whole_inactive_records_only(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    await _seed(store, "P-TEST-OLD")
    await _seed(store, "P-TEST-ACTIVE")
    await _backdate(store, "P-TEST-OLD", _NOW - timedelta(days=4000))
    await _backdate(store, "P-TEST-ACTIVE", _NOW - timedelta(days=4000))
    # A recent visit keeps an otherwise old record alive.
    async with store.connect() as conn:
        await conn.execute(
            "UPDATE visits SET visit_date = $1 WHERE patient_id = 'P-TEST-ACTIVE'",
            _NOW - timedelta(days=10),
        )

    result = await purge_expired(
        store, retention_days=3650, audit_retention_days=None, execute=True, now=_NOW
    )

    assert result.patients == 1
    for table in ("medications", "lab_results", "visits"):
        assert await _count(store, table, "P-TEST-OLD") == 0
        assert await _count(store, table, "P-TEST-ACTIVE") == 1
    assert await PatientStore(store).get_patient("P-TEST-OLD") is None
    await store.close()


async def test_audit_entries_purged_by_age_and_the_purge_is_audited(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    audit = AuditLogStore(store)
    await audit.record(_USER, "P-TEST-1", "patient_viewed")
    await audit.record(_USER, "P-TEST-1", "patient_viewed")
    async with store.connect() as conn:
        await conn.execute(
            "UPDATE audit_log SET created_at = $1 WHERE id = (SELECT id FROM audit_log LIMIT 1)",
            _NOW - timedelta(days=4000),
        )

    result = await purge_expired(
        store, retention_days=None, audit_retention_days=3650, execute=True, now=_NOW
    )

    assert result.audit_entries == 1
    async with store.connect() as conn:
        actions = [r["action"] for r in await conn.fetch("SELECT action FROM audit_log")]
    assert sorted(actions) == ["patient_viewed", "retention_purge"]
    await store.close()


async def test_unconfigured_retention_does_nothing(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    await _seed(store, "P-TEST-OLD")
    await _backdate(store, "P-TEST-OLD", _NOW - timedelta(days=40000))

    result = await purge_expired(
        store, retention_days=None, audit_retention_days=None, execute=True, now=_NOW
    )

    assert (result.patients, result.audit_entries) == (None, None)
    assert await _count(store, "visits", "P-TEST-OLD") == 1
    await store.close()


@pytest.mark.parametrize("field", ["retention_days", "audit_retention_days"])
def test_retention_under_a_year_is_refused(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 30})

"""Retention: delete patient records and audit entries older than configured.

Opt-in and conservative. Nothing is deleted unless RETENTION_DAYS /
AUDIT_RETENTION_DAYS are set; a dry run (the default) only counts. A patient
record expires as a whole -- patient, medications, labs, visits, interactions
-- once its last activity (the record's update time or newest visit) is older
than the cutoff; deleting part of a record would corrupt its history. Audit
entries are kept independently (they must outlive the data they describe),
and every purge writes an audit entry of its own.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from medagent.memory.audit_log import AuditLogStore
from medagent.memory.persistent import PersistentStore

_EXPIRED_PATIENTS = """
    SELECT p.id FROM patients p
    LEFT JOIN LATERAL (
        SELECT max(visit_date) AS last_visit FROM visits v WHERE v.patient_id = p.id
    ) lv ON TRUE
    WHERE greatest(p.updated_at, coalesce(lv.last_visit, p.updated_at)) < $1
"""
# Children first: they reference patients(id).
_PATIENT_TABLES = ("interactions_log", "visits", "lab_results", "medications")


@dataclass
class PurgeResult:
    patients: int | None  # None = record retention not configured
    audit_entries: int | None  # None = audit retention not configured
    executed: bool


async def purge_expired(
    store: PersistentStore,
    *,
    retention_days: int | None,
    audit_retention_days: int | None,
    execute: bool,
    now: datetime | None = None,
) -> PurgeResult:
    now = now or datetime.now(UTC)
    patients: int | None = None
    audit_entries: int | None = None

    async with store.connect() as conn:
        if retention_days is not None:
            cutoff = now - timedelta(days=retention_days)
            async with conn.transaction():
                ids = [row["id"] for row in await conn.fetch(_EXPIRED_PATIENTS, cutoff)]
                if execute and ids:
                    for table in _PATIENT_TABLES:
                        await conn.execute(
                            f"DELETE FROM {table} WHERE patient_id = ANY($1::text[])", ids
                        )
                    await conn.execute("DELETE FROM patients WHERE id = ANY($1::text[])", ids)
            patients = len(ids)

        if audit_retention_days is not None:
            cutoff = now - timedelta(days=audit_retention_days)
            if execute:
                status = await conn.execute("DELETE FROM audit_log WHERE created_at < $1", cutoff)
                audit_entries = int(status.split()[-1])
            else:
                audit_entries = await conn.fetchval(
                    "SELECT count(*) FROM audit_log WHERE created_at < $1", cutoff
                )

    if execute and (patients is not None or audit_entries is not None):
        # Written after the audit purge, so the record of the purge survives it.
        await AuditLogStore(store).record(
            None,
            None,
            "retention_purge",
            {
                "patients_deleted": patients,
                "audit_entries_deleted": audit_entries,
                "retention_days": retention_days,
                "audit_retention_days": audit_retention_days,
            },
        )
    return PurgeResult(patients=patients, audit_entries=audit_entries, executed=execute)

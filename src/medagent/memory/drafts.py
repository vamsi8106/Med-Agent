"""Approval drafts held in Postgres (STATE_BACKEND=postgres).

LangGraph's AsyncPostgresSaver owns reading and writing the checkpoint_* tables
(workflows/checkpointing.py). This module only deletes from them: drafts are
patient data, so abandoned ones expire after APPROVAL_DRAFT_TTL_MINUTES and a
retention purge takes a patient's drafts with the rest of the record.

Thread ids are "{user_id}:{patient_id}" (api/routes/ws.py). User ids are UUIDs,
so the patient id is everything after the first colon.
"""

from datetime import UTC, datetime, timedelta

import asyncpg

from medagent.memory.persistent import PersistentStore

# Children before parents isn't required (no foreign keys), but keep one order.
_DRAFT_TABLES = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")

_EXPIRED_THREADS = """
    SELECT thread_id FROM checkpoints
    GROUP BY thread_id
    HAVING max((checkpoint->>'ts')::timestamptz) < $1
"""


async def _delete_threads(conn: asyncpg.Connection, thread_ids: list[str]) -> None:
    for table in _DRAFT_TABLES:
        await conn.execute(f"DELETE FROM {table} WHERE thread_id = ANY($1::text[])", thread_ids)


async def purge_expired_drafts(
    store: PersistentStore, ttl: timedelta, now: datetime | None = None
) -> int:
    """Deletes every draft whose newest checkpoint is older than `ttl`; returns
    how many were removed. Safe to run from several replicas at once."""
    cutoff = (now or datetime.now(UTC)) - ttl
    async with store.connect() as conn, conn.transaction():
        expired = [row["thread_id"] for row in await conn.fetch(_EXPIRED_THREADS, cutoff)]
        if expired:
            await _delete_threads(conn, expired)
    return len(expired)


async def delete_drafts_for_patients(conn: asyncpg.Connection, patient_ids: list[str]) -> None:
    """Every doctor's drafts for these patients. Runs on the caller's
    connection so retention deletes drafts in the same transaction as the record."""
    rows = await conn.fetch(
        "SELECT DISTINCT thread_id FROM checkpoints "
        "WHERE split_part(thread_id, ':', 2) = ANY($1::text[])",
        patient_ids,
    )
    if rows:
        await _delete_threads(conn, [row["thread_id"] for row in rows])

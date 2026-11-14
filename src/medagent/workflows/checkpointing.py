"""Checkpointers for paused follow-up drafts.

A checkpoint holds the full PatientContext and the draft report, i.e. patient
data. Two stores, chosen by STATE_BACKEND:

- memory (default): process memory. A restart loses pending drafts, and it
  only works with a single replica.
- postgres: LangGraph's AsyncPostgresSaver on the checkpoint_* tables
  (migration 0006), so any replica can resume a draft and a restart keeps it.
  Drafts are scoped by thread id ("{user_id}:{patient_id}"), expire after
  APPROVAL_DRAFT_TTL_MINUTES (memory/drafts.py), are deleted with their
  patient by retention, and rely on the database's encryption at rest.

LangGraph deserializes checkpoint state through an allowlist of types and warns
that unregistered ones "will be blocked in a future version", so every type
that can appear in the follow-up/assessment state is registered explicitly.
"""

from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from medagent.core.exceptions import MemoryError as MedAgentMemoryError
from medagent.infra.logging import get_logger

logger = get_logger(__name__)

_CHECKPOINT_STATE_TYPES: list[tuple[str, str]] = [
    ("medagent.core.models", "PatientContext"),
    ("medagent.core.models", "Medication"),
    ("medagent.core.models", "LabResult"),
    ("medagent.core.models", "Visit"),
    ("medagent.core.models", "RecordChanges"),
    ("medagent.core.models", "MedicationChange"),
    ("medagent.core.models", "LabChange"),
    ("medagent.core.models", "AgentResult"),
    ("medagent.core.models", "AgentFailure"),
    ("medagent.core.models", "ClinicalEvidence"),
    ("medagent.core.models", "DrugInteraction"),
    ("medagent.core.types", "AgentRole"),
    ("medagent.core.types", "InteractionSeverity"),
    ("medagent.core.types", "EvidenceGrade"),
    ("medagent.infra.agent_run", "AgentRunTracker"),
    ("medagent.infra.agent_run", "AgentStepRecord"),
]


def build_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_STATE_TYPES)


def build_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=build_serializer())


async def open_postgres_checkpointer(
    dsn: str,
) -> tuple[AsyncPostgresSaver, AsyncConnectionPool[AsyncConnection[DictRow]]]:
    """A Postgres-backed checkpointer and the pool it runs on (close the pool on
    shutdown). The tables come from Alembic, never from saver.setup(): refuses
    to start if the installed saver expects schema versions the database lacks."""
    # autocommit / prepare_threshold / dict_row: what AsyncPostgresSaver requires
    # of its connections (as in its own from_conn_string).
    pool = AsyncConnectionPool(
        dsn,
        connection_class=AsyncConnection[DictRow],
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    try:
        async with pool.connection() as conn:
            cursor = await conn.execute("SELECT max(v) AS v FROM checkpoint_migrations")
            row: DictRow | None = await cursor.fetchone()
    except Exception:
        await pool.close()
        raise
    expected = len(AsyncPostgresSaver.MIGRATIONS) - 1
    if row is None or row["v"] != expected:
        await pool.close()
        raise MedAgentMemoryError(
            f"Checkpoint schema is at version {row['v'] if row else None}, the installed "
            f"langgraph-checkpoint-postgres expects {expected}: add an Alembic migration "
            "(see CHECKPOINT_SAVER_MIGRATIONS in memory/schema.py)"
        )
    return AsyncPostgresSaver(pool, serde=build_serializer()), pool


async def purge_expired_drafts(
    saver: BaseCheckpointSaver, ttl: timedelta, now: datetime | None = None
) -> int:
    """Deletes every thread whose latest checkpoint is older than `ttl`, so
    drafts abandoned by a dropped connection don't hold patient data in memory
    indefinitely. Returns how many threads were removed."""
    current = now or datetime.now(UTC)
    latest: dict[str, datetime] = {}
    async for checkpoint_tuple in saver.alist(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        written = datetime.fromisoformat(checkpoint_tuple.checkpoint["ts"])
        if thread_id not in latest or written > latest[thread_id]:
            latest[thread_id] = written

    expired = [thread for thread, written in latest.items() if current - written > ttl]
    for thread_id in expired:
        await saver.adelete_thread(thread_id)
    if expired:
        logger.info("approval_drafts_purged", count=len(expired))
    return len(expired)

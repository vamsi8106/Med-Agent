import json
from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from medagent.memory.drafts import delete_drafts_for_patients, purge_expired_drafts
from medagent.memory.persistent import PersistentStore
from medagent.memory.schema import CHECKPOINT_SAVER_MIGRATIONS

_NOW = datetime(2026, 6, 1, tzinfo=UTC)


def _normalized(sql: str) -> str:
    return " ".join(sql.replace("CONCURRENTLY ", "").split()).rstrip(";")


def test_frozen_checkpoint_schema_matches_the_installed_saver() -> None:
    """Fails when a langgraph-checkpoint-postgres upgrade adds or changes a
    migration: copy it into CHECKPOINT_SAVER_MIGRATIONS in a new Alembic
    migration (the app also refuses to start until then)."""
    assert [_normalized(s) for s in CHECKPOINT_SAVER_MIGRATIONS] == [
        _normalized(s) for s in AsyncPostgresSaver.MIGRATIONS
    ]


async def _add_draft(store: PersistentStore, thread_id: str, written: datetime) -> None:
    async with store.connect() as conn:
        await conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_id, checkpoint) VALUES ($1, $2, $3)",
            thread_id,
            f"cp-{written.isoformat()}",
            json.dumps({"ts": written.isoformat()}),
        )
        await conn.execute(
            "INSERT INTO checkpoint_blobs (thread_id, channel, version, type) "
            "VALUES ($1, 'report', '1', 'msgpack') ON CONFLICT DO NOTHING",
            thread_id,
        )


async def _threads(store: PersistentStore) -> set[str]:
    async with store.connect() as conn:
        rows = await conn.fetch(
            "SELECT thread_id FROM checkpoints UNION SELECT thread_id FROM checkpoint_blobs"
        )
    return {row["thread_id"] for row in rows}


async def test_purge_removes_only_drafts_older_than_the_ttl(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    await _add_draft(store, "U1:P-TEST-1", _NOW - timedelta(hours=2))
    await _add_draft(store, "U1:P-TEST-2", _NOW - timedelta(minutes=5))
    # A thread with one old and one recent checkpoint is still in use.
    await _add_draft(store, "U2:P-TEST-3", _NOW - timedelta(hours=2))
    await _add_draft(store, "U2:P-TEST-3", _NOW - timedelta(minutes=1))

    removed = await purge_expired_drafts(store, timedelta(minutes=30), now=_NOW)

    assert removed == 1
    assert await _threads(store) == {"U1:P-TEST-2", "U2:P-TEST-3"}
    await store.close()


async def test_a_patients_drafts_are_deleted_for_every_doctor(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    await _add_draft(store, "U1:P-TEST-1", _NOW)
    await _add_draft(store, "U2:P-TEST-1", _NOW)
    await _add_draft(store, "U1:P-TEST-10", _NOW)  # a prefix match must not count

    async with store.connect() as conn:
        await delete_drafts_for_patients(conn, ["P-TEST-1"])

    assert await _threads(store) == {"U1:P-TEST-10"}
    await store.close()

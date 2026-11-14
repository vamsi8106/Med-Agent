import pytest

from medagent.core.exceptions import MedAgentError
from medagent.memory.persistent import PersistentStore
from medagent.memory.schema import EMBEDDING_DIMENSIONS as SCHEMA_DIMENSIONS
from medagent.rag.vector_store import EMBEDDING_DIMENSIONS, StoredChunk, VectorStore


def _vec(*leading: float) -> list[float]:
    """A 384-dim vector whose first components are `leading`."""
    return [*leading, *([0.0] * (EMBEDDING_DIMENSIONS - len(leading)))]


def _chunk(position: int, text: str, vector: list[float], section: str = "3.4 BP") -> StoredChunk:
    return StoredChunk(
        id=f"{position}-{text[:8]}", position=position, section=section, text=text, embedding=vector
    )


async def _store_with(pg_dsn: str, chunks: list[StoredChunk], source: str = "kdigo") -> VectorStore:
    store = VectorStore(PersistentStore(pg_dsn))
    await store.replace_document(
        source, content_hash="h1", embedding_model="m", chunker_version="v1", chunks=chunks
    )
    return store


def test_the_store_and_the_schema_agree_on_the_vector_size() -> None:
    assert EMBEDDING_DIMENSIONS == SCHEMA_DIMENSIONS


async def test_query_returns_nearest_chunks_first(pg_dsn: str) -> None:
    store = await _store_with(
        pg_dsn,
        [
            _chunk(0, "Target systolic blood pressure below 120.", _vec(1.0, 0.0)),
            _chunk(1, "Avoid tobacco products.", _vec(0.0, 1.0)),
        ],
    )

    results = await store.query(_vec(0.9, 0.1), top_k=2)

    assert [r["document"] for r in results] == [
        "Target systolic blood pressure below 120.",
        "Avoid tobacco products.",
    ]
    assert results[0]["metadata"] == {"source": "kdigo", "section": "3.4 BP"}
    assert results[0]["distance"] < results[1]["distance"]


async def test_keyword_query_matches_any_word_and_ranks(pg_dsn: str) -> None:
    store = await _store_with(
        pg_dsn,
        [
            _chunk(0, "Start an SGLT2 inhibitor in CKD with albuminuria.", _vec(1.0)),
            _chunk(1, "Statins lower cardiovascular risk.", _vec(0.0, 1.0)),
            _chunk(2, "Exercise 150 minutes per week.", _vec(0.0, 0.0, 1.0)),
        ],
    )

    results = await store.keyword_query("Which patients need an SGLT2 inhibitor?", top_k=5)

    assert [r["document"] for r in results] == ["Start an SGLT2 inhibitor in CKD with albuminuria."]
    assert await store.keyword_query("the of and", top_k=5) == []  # only stop words


async def test_replace_document_keeps_unchanged_chunks_and_swaps_the_rest(pg_dsn: str) -> None:
    kept = _chunk(0, "Unchanged recommendation.", _vec(1.0))
    store = await _store_with(pg_dsn, [kept, _chunk(1, "Old wording.", _vec(0.0, 1.0))])
    async with PersistentStore(pg_dsn).connect() as conn:
        await conn.execute("UPDATE guideline_chunks SET section = 'marker' WHERE id = $1", kept.id)

    await store.replace_document(
        "kdigo",
        content_hash="h2",
        embedding_model="m",
        chunker_version="v1",
        chunks=[kept, _chunk(1, "New wording.", _vec(0.0, 1.0))],
    )

    async with PersistentStore(pg_dsn).connect() as conn:
        rows = await conn.fetch("SELECT id, section, text FROM guideline_chunks ORDER BY position")
    # The unchanged chunk was not rewritten (still carries the marker).
    assert [(r["section"], r["text"]) for r in rows] == [
        ("marker", "Unchanged recommendation."),
        ("3.4 BP", "New wording."),
    ]
    version = await store.document("kdigo")
    assert version is not None
    assert (version.content_hash, version.chunk_count) == ("h2", 2)


async def test_documents_are_independent(pg_dsn: str) -> None:
    store = await _store_with(pg_dsn, [_chunk(0, "KDIGO text.", _vec(1.0))], source="kdigo")
    await store.replace_document(
        "nice",
        content_hash="n",
        embedding_model="m",
        chunker_version="v1",
        chunks=[_chunk(0, "NICE text.", _vec(0.0, 1.0))],
    )

    await store.replace_document(
        "kdigo", content_hash="h3", embedding_model="m", chunker_version="v1", chunks=[]
    )

    assert [r["document"] for r in await store.query(_vec(1.0), top_k=5)] == ["NICE text."]
    assert await store.document("missing") is None


async def test_a_vector_of_the_wrong_size_is_refused(pg_dsn: str) -> None:
    store = VectorStore(PersistentStore(pg_dsn))
    with pytest.raises(MedAgentError, match="needs a migration"):
        await store.query([0.1, 0.2], top_k=1)

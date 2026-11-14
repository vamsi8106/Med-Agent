import chromadb

from medagent.rag.vector_store import VectorStore


def _store(name: str) -> VectorStore:
    # In-memory client: network-free, but exercises real chromadb add/query/delete.
    return VectorStore("unused", 0, name, client=chromadb.EphemeralClient())


async def test_add_and_query_round_trips() -> None:
    store = _store("test-guidelines")

    await store.add(
        ids=["1", "2"],
        documents=["Metformin is first-line therapy for type 2 diabetes.", "Unrelated text."],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"source": "kdigo-guideline"}, {"source": "other"}],
    )

    results = await store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "kdigo-guideline"


async def test_delete_source_keeps_the_new_ingestion_and_other_sources() -> None:
    store = _store("test-delete-source")
    await store.add(
        ids=["old", "new", "other"],
        documents=["old chunk", "new chunk", "other document"],
        embeddings=[[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]],
        metadatas=[
            {"source": "kdigo", "ingestion_id": "first"},
            {"source": "kdigo", "ingestion_id": "second"},
            {"source": "nice", "ingestion_id": "first"},
        ],
    )

    await store.delete_source("kdigo", except_ingestion="second")

    results = await store.query(query_embedding=[1.0, 0.0], top_k=3)
    assert sorted(r["document"] for r in results) == ["new chunk", "other document"]

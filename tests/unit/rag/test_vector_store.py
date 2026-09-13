from pathlib import Path

from medagent.rag.vector_store import VectorStore


async def test_add_and_query_round_trips(tmp_path: Path) -> None:
    store = VectorStore(persist_dir=str(tmp_path), collection_name="test-guidelines")

    await store.add(
        ids=["1", "2"],
        documents=["Metformin is first-line therapy for type 2 diabetes.", "Unrelated text."],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"source": "ada-guideline"}, {"source": "other"}],
    )

    results = await store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "ada-guideline"

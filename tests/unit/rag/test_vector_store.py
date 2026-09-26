from unittest.mock import patch

import chromadb

from medagent.rag.vector_store import VectorStore


async def test_add_and_query_round_trips() -> None:
    # VectorStore always constructs chromadb.HttpClient (production talks to the
    # chromadb server container) -- redirect it to an in-memory client here so
    # this stays a network-free unit test while still exercising real add/query
    # logic, not a mock.
    with patch(
        "medagent.rag.vector_store.chromadb.HttpClient", return_value=chromadb.EphemeralClient()
    ):
        store = VectorStore(host="unused", port=0, collection_name="test-guidelines")

    await store.add(
        ids=["1", "2"],
        documents=["Metformin is first-line therapy for type 2 diabetes.", "Unrelated text."],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"source": "ada-guideline"}, {"source": "other"}],
    )

    results = await store.query(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "ada-guideline"

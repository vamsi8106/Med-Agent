from unittest.mock import AsyncMock

from medagent.rag.retriever import GuidelineRetrieverTool


async def test_retriever_returns_clinical_evidence() -> None:
    embeddings = AsyncMock()
    embeddings.embed.return_value = [[0.1, 0.2]]
    vector_store = AsyncMock()
    vector_store.query.return_value = [
        {
            "document": "Metformin is first-line therapy for T2DM.",
            "metadata": {"source": "ada-2024", "section": "Pharmacotherapy"},
            "distance": 0.1,
        }
    ]

    tool = GuidelineRetrieverTool(embeddings=embeddings, vector_store=vector_store)
    results = await tool.run(query="first-line therapy for diabetes", top_k=3)

    assert len(results) == 1
    assert results[0].title == "Pharmacotherapy"
    assert results[0].source == "ada-2024"
    vector_store.query.assert_awaited_once_with([0.1, 0.2], top_k=3)

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


async def test_with_a_reranker_it_fetches_candidates_and_returns_the_reranked_top_k() -> None:
    embeddings = AsyncMock()
    embeddings.embed.return_value = [[0.1, 0.2]]
    vector_store = AsyncMock()
    vector_store.query.return_value = [
        {"document": f"chunk {i}", "metadata": {"source": "kdigo", "section": "3.4"}, "distance": i}
        for i in range(20)
    ]
    reranker = AsyncMock()
    reranker.rerank.side_effect = lambda query, hits, top_n: list(reversed(hits))[:top_n]

    tool = GuidelineRetrieverTool(embeddings, vector_store, reranker=reranker, candidates=20)
    evidence = await tool.run(query="bp target", top_k=3)

    vector_store.query.assert_awaited_once_with([0.1, 0.2], top_k=20)
    assert [e.summary for e in evidence] == ["chunk 19", "chunk 18", "chunk 17"]
    assert reranker.rerank.call_args.args[0] == "bp target"


async def test_without_a_reranker_it_fetches_only_top_k() -> None:
    embeddings = AsyncMock()
    embeddings.embed.return_value = [[0.1, 0.2]]
    vector_store = AsyncMock()
    vector_store.query.return_value = []

    await GuidelineRetrieverTool(embeddings, vector_store, candidates=20).run(query="q", top_k=3)

    vector_store.query.assert_awaited_once_with([0.1, 0.2], top_k=3)

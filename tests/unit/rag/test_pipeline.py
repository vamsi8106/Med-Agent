from unittest.mock import AsyncMock, MagicMock

from medagent.rag.pipeline import IngestionPipeline


def _embeddings(vectors: list[list[float]]) -> MagicMock:
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(return_value=vectors)
    embeddings.max_tokens = 254
    embeddings.model_name = "all-MiniLM-L6-v2"
    embeddings.count_tokens = lambda words: [1] * len(words)
    return embeddings


async def test_ingest_text_embeds_and_stores_chunks() -> None:
    embeddings = _embeddings([[0.1, 0.2]])
    vector_store = AsyncMock()

    pipeline = IngestionPipeline(embeddings=embeddings, vector_store=vector_store)

    count = await pipeline.ingest_text("# Dosing\nGuideline body text.", source_name="kdigo-2024")

    assert count == 1
    # The embedding sees the context prefix; the store keeps the plain text.
    [embedded] = embeddings.embed.call_args.args[0]
    assert embedded == "kdigo-2024 › Dosing\nGuideline body text."
    _, kwargs = vector_store.add.call_args
    assert kwargs["documents"] == ["Guideline body text."]
    metadata = kwargs["metadatas"][0]
    assert metadata["source"] == "kdigo-2024"
    assert metadata["section"] == "Dosing"
    assert metadata["embedding_model"] == "all-MiniLM-L6-v2"


async def test_reingesting_replaces_the_previous_chunks_after_adding() -> None:
    embeddings = _embeddings([[0.1, 0.2]])
    vector_store = AsyncMock()
    calls: list[str] = []
    vector_store.add.side_effect = lambda **_: calls.append("add")
    vector_store.delete_source.side_effect = lambda *_, **__: calls.append("delete")

    await IngestionPipeline(embeddings, vector_store).ingest_text("Body.", source_name="kdigo")

    assert calls == ["add", "delete"]  # never a moment with the document missing
    ingestion_id = vector_store.add.call_args.kwargs["metadatas"][0]["ingestion_id"]
    vector_store.delete_source.assert_awaited_once_with("kdigo", except_ingestion=ingestion_id)


async def test_ingest_text_empty_returns_zero() -> None:
    embeddings = _embeddings([])
    vector_store = AsyncMock()
    pipeline = IngestionPipeline(embeddings=embeddings, vector_store=vector_store)

    count = await pipeline.ingest_text("   ", source_name="empty")

    assert count == 0
    embeddings.embed.assert_not_called()

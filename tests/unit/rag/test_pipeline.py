from unittest.mock import AsyncMock

from medagent.rag.pipeline import IngestionPipeline


async def test_ingest_text_embeds_and_stores_chunks() -> None:
    embeddings = AsyncMock()
    embeddings.embed.return_value = [[0.1, 0.2]]
    vector_store = AsyncMock()

    pipeline = IngestionPipeline(embeddings=embeddings, vector_store=vector_store)

    count = await pipeline.ingest_text("Guideline body text.", source_name="ada-2024")

    assert count == 1
    vector_store.add.assert_awaited_once()
    _, kwargs = vector_store.add.call_args
    assert kwargs["metadatas"][0]["source"] == "ada-2024"


async def test_ingest_text_empty_returns_zero() -> None:
    embeddings = AsyncMock()
    vector_store = AsyncMock()
    pipeline = IngestionPipeline(embeddings=embeddings, vector_store=vector_store)

    count = await pipeline.ingest_text("   ", source_name="empty")

    assert count == 0
    embeddings.embed.assert_not_called()

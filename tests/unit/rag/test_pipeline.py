from unittest.mock import AsyncMock, MagicMock

from medagent.rag.chunker import CHUNKER_VERSION
from medagent.rag.pipeline import EMBED_BATCH_SIZE, IngestionPipeline, chunk_id
from medagent.rag.vector_store import DocumentVersion


def _embeddings() -> MagicMock:
    embeddings = MagicMock()
    embeddings.embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])
    embeddings.max_tokens = 254
    embeddings.model_name = "all-MiniLM-L6-v2"
    embeddings.count_tokens = lambda words: [1] * len(words)
    return embeddings


def _store(stored: DocumentVersion | None = None) -> AsyncMock:
    store = AsyncMock()
    store.document.return_value = stored
    return store


async def test_ingest_embeds_with_context_and_stores_plain_text_with_stable_ids() -> None:
    embeddings, store = _embeddings(), _store()

    result = await IngestionPipeline(embeddings, store).ingest_text(
        "# Dosing\nGuideline body text.", source_name="kdigo-2024"
    )

    assert (result.chunks, result.skipped) == (1, False)
    [embedded] = embeddings.embed.call_args.args[0]
    assert embedded == "kdigo-2024 › Dosing\nGuideline body text."
    kwargs = store.replace_document.call_args.kwargs
    assert store.replace_document.call_args.args == ("kdigo-2024",)
    assert kwargs["embedding_model"] == "all-MiniLM-L6-v2"
    assert kwargs["chunker_version"] == CHUNKER_VERSION
    [chunk] = kwargs["chunks"]
    assert (chunk.text, chunk.section, chunk.position) == ("Guideline body text.", "Dosing", 0)
    assert chunk.id == chunk_id("kdigo-2024", 0, "Guideline body text.")


def test_chunk_ids_are_stable_and_content_addressed() -> None:
    assert chunk_id("kdigo", 3, "text") == chunk_id("kdigo", 3, "text")
    assert chunk_id("kdigo", 3, "text") != chunk_id("kdigo", 3, "other text")
    assert chunk_id("kdigo", 3, "text") != chunk_id("nice", 3, "text")


async def test_an_unchanged_document_is_not_re_embedded() -> None:
    import hashlib

    text = "Guideline body text."
    stored = DocumentVersion(
        source="kdigo",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        embedding_model="all-MiniLM-L6-v2",
        chunker_version=CHUNKER_VERSION,
        chunk_count=7,
        ingested_at=MagicMock(),
    )
    embeddings, store = _embeddings(), _store(stored)
    pipeline = IngestionPipeline(embeddings, store)

    result = await pipeline.ingest_text(text, source_name="kdigo")

    assert (result.chunks, result.skipped) == (7, True)
    embeddings.embed.assert_not_called()
    store.replace_document.assert_not_called()

    forced = await pipeline.ingest_text(text, source_name="kdigo", force=True)
    assert forced.skipped is False
    store.replace_document.assert_awaited_once()


async def test_a_new_chunker_or_model_re_ingests() -> None:
    stored = DocumentVersion("kdigo", "same", "old-model", CHUNKER_VERSION, 1, MagicMock())
    store = _store(stored)
    store.document.return_value = stored

    result = await IngestionPipeline(_embeddings(), store).ingest_text("Body.", source_name="kdigo")

    assert result.skipped is False


async def test_embedding_runs_in_batches() -> None:
    embeddings, store = _embeddings(), _store()
    text = "\n".join(
        f"# Section {i}\nBody of section number {i}." for i in range(EMBED_BATCH_SIZE + 10)
    )

    result = await IngestionPipeline(embeddings, store).ingest_text(text, source_name="big")

    assert result.chunks == EMBED_BATCH_SIZE + 10
    assert [len(call.args[0]) for call in embeddings.embed.call_args_list] == [EMBED_BATCH_SIZE, 10]


async def test_ingest_text_empty_returns_zero() -> None:
    embeddings, store = _embeddings(), _store()

    result = await IngestionPipeline(embeddings, store).ingest_text("   ", source_name="empty")

    assert result.chunks == 0
    embeddings.embed.assert_not_called()

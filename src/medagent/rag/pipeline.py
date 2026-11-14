"""Guideline ingestion pipeline: PDF -> chunk -> embed -> store.

Versioned and idempotent: a document's version is the hash of its extracted
text plus the embedding model and chunker version that produced its chunks.
Ingesting an unchanged document does nothing; a changed one replaces its
chunks in one transaction, and chunks whose text did not change keep their ids
(sha256 of source, position and text). Embedding runs in batches, so a large
document never builds one huge batch in memory.
"""

import hashlib
from dataclasses import dataclass

from pypdf import PdfReader

from medagent.core.exceptions import MedAgentError
from medagent.infra.logging import get_logger
from medagent.rag.chunker import CHUNK_OVERLAP_TOKENS, CHUNKER_VERSION, chunk_text
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.vector_store import StoredChunk, VectorStore

logger = get_logger(__name__)

EMBED_BATCH_SIZE = 256


def _extract_pdf_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise MedAgentError(f"Failed to read PDF at {pdf_path}: {exc}") from exc
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_id(source: str, position: int, text: str) -> str:
    return hashlib.sha256(f"{source}\0{position}\0{text}".encode()).hexdigest()[:32]


@dataclass(frozen=True)
class IngestResult:
    chunks: int
    skipped: bool  # unchanged since the last ingestion: nothing embedded or written


class IngestionPipeline:
    def __init__(self, embeddings: EmbeddingModel, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def ingest_pdf(
        self, pdf_path: str, source_name: str, *, force: bool = False
    ) -> IngestResult:
        text = _extract_pdf_text(pdf_path)
        return await self.ingest_text(text, source_name, force=force)

    async def ingest_text(
        self, text: str, source_name: str, *, force: bool = False
    ) -> IngestResult:
        """Chunks to the embedding model's limit (reference lists left out),
        embeds each chunk with its "<source> › <section>" context, stores the
        plain chunk text. Skipped when this exact text was already ingested
        with the same model and chunker, unless `force`. Measured by
        `make rag-eval`."""
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        stored = await self._vector_store.document(source_name)
        if (
            not force
            and stored is not None
            and (stored.content_hash, stored.embedding_model, stored.chunker_version)
            == (content_hash, self._embeddings.model_name, CHUNKER_VERSION)
        ):
            logger.info("guideline_unchanged", source=source_name, chunk_count=stored.chunk_count)
            return IngestResult(chunks=stored.chunk_count, skipped=True)

        chunks = chunk_text(
            text,
            self._embeddings.max_tokens,
            CHUNK_OVERLAP_TOKENS,
            count_tokens=self._embeddings.count_tokens,
            title=source_name,
            drop_reference_lists=True,
        )
        if not chunks:
            return IngestResult(chunks=0, skipped=False)

        vectors: list[list[float]] = []
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors.extend(await self._embeddings.embed([c.embedding_text for c in batch]))

        await self._vector_store.replace_document(
            source_name,
            content_hash=content_hash,
            # Vectors from different models are not comparable: a model change
            # re-embeds every document (the version check above sees it).
            embedding_model=self._embeddings.model_name,
            chunker_version=CHUNKER_VERSION,
            chunks=[
                StoredChunk(
                    id=chunk_id(source_name, position, chunk.text),
                    position=position,
                    section=chunk.section,
                    text=chunk.text,
                    embedding=vector,
                )
                for position, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
            ],
        )
        logger.info("guideline_ingested", source=source_name, chunk_count=len(chunks))
        return IngestResult(chunks=len(chunks), skipped=False)

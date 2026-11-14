"""Guideline ingestion pipeline: PDF -> chunk -> embed -> store."""

import uuid

from pypdf import PdfReader

from medagent.core.exceptions import MedAgentError
from medagent.infra.logging import get_logger
from medagent.rag.chunker import CHUNK_OVERLAP_TOKENS, chunk_text
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.vector_store import VectorStore

logger = get_logger(__name__)


def _extract_pdf_text(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:
        raise MedAgentError(f"Failed to read PDF at {pdf_path}: {exc}") from exc
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class IngestionPipeline:
    def __init__(self, embeddings: EmbeddingModel, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def ingest_pdf(self, pdf_path: str, source_name: str) -> int:
        text = _extract_pdf_text(pdf_path)
        return await self.ingest_text(text, source_name)

    async def ingest_text(self, text: str, source_name: str) -> int:
        """Chunks to the embedding model's limit (reference lists left out),
        embeds each chunk with its "<source> › <section>" context, stores the
        plain chunk text, then drops the source's chunks from any earlier
        ingestion (re-ingesting replaces). Measured by `make rag-eval`."""
        chunks = chunk_text(
            text,
            self._embeddings.max_tokens,
            CHUNK_OVERLAP_TOKENS,
            count_tokens=self._embeddings.count_tokens,
            title=source_name,
            drop_reference_lists=True,
        )
        if not chunks:
            return 0

        embeddings = await self._embeddings.embed([c.embedding_text for c in chunks])
        ingestion_id = str(uuid.uuid4())
        metadatas = [
            {
                "source": source_name,
                "section": c.section or "",
                "ingestion_id": ingestion_id,
                # Vectors from different models are not comparable: a model
                # change means re-ingesting every document.
                "embedding_model": self._embeddings.model_name,
            }
            for c in chunks
        ]

        await self._vector_store.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        await self._vector_store.delete_source(source_name, except_ingestion=ingestion_id)
        logger.info("guideline_ingested", source=source_name, chunk_count=len(chunks))
        return len(chunks)

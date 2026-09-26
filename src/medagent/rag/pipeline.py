"""Guideline ingestion pipeline: PDF -> chunk -> embed -> store."""

import uuid

from pypdf import PdfReader

from medagent.core.exceptions import MedAgentError
from medagent.infra.logging import get_logger
from medagent.rag.chunker import chunk_text
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
        chunks = chunk_text(text)
        if not chunks:
            return 0

        embeddings = await self._embeddings.embed([c.text for c in chunks])
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": source_name, "section": c.section or ""} for c in chunks]

        await self._vector_store.add(
            ids=ids,
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("guideline_ingested", source=source_name, chunk_count=len(chunks))
        return len(chunks)

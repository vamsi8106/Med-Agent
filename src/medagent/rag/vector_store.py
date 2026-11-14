"""Guideline chunks in Postgres with pgvector: similarity, keyword and versioned writes.

Replaces the separate ChromaDB server: the vectors live next to everything else
(one database to secure, back up and scale; RDS supports the extension), and
Postgres full-text search gives hybrid keyword + vector retrieval for free.
Tables come from migration 0007 (memory/schema.py).

Vectors travel as text literals ('[0.1,0.2,...]'::vector), so no extra driver
codec is needed. `rag/` may not import `memory/` (layer rule): the connection
source is any object with PersistentStore's `connect()`.
"""

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import asyncpg

from medagent.core.exceptions import MedAgentError

# Must match guideline_chunks.embedding (memory/schema.py EMBEDDING_DIMENSIONS).
EMBEDDING_DIMENSIONS = 384


class ConnectionSource(Protocol):
    def connect(self) -> AbstractAsyncContextManager[asyncpg.Connection]: ...


@dataclass(frozen=True)
class StoredChunk:
    id: str
    position: int
    section: str | None
    text: str
    embedding: list[float]


@dataclass(frozen=True)
class DocumentVersion:
    source: str
    content_hash: str
    embedding_model: str
    chunker_version: str
    chunk_count: int
    ingested_at: datetime


def _vector(embedding: list[float]) -> str:
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise MedAgentError(
            f"Embedding has {len(embedding)} dimensions, guideline_chunks stores "
            f"{EMBEDDING_DIMENSIONS}: a different embedding model needs a migration"
        )
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _match(row: asyncpg.Record, distance: float) -> dict[str, Any]:
    return {
        "document": row["text"],
        "metadata": {"source": row["source"], "section": row["section"] or ""},
        "distance": distance,
    }


class VectorStore:
    def __init__(self, connections: ConnectionSource) -> None:
        self._connections = connections

    async def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        """Nearest chunks by cosine distance (HNSW index)."""
        async with self._connections.connect() as conn:
            rows = await conn.fetch(
                """SELECT source, section, text, embedding <=> $1::vector AS distance
                FROM guideline_chunks ORDER BY embedding <=> $1::vector LIMIT $2""",
                _vector(query_embedding),
                top_k,
            )
        return [_match(row, row["distance"]) for row in rows]

    async def keyword_query(self, text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Chunks containing any of the query's words (English stemming, stop
        words dropped), best ts_rank_cd first. plainto_tsquery joins words with
        AND, which a natural question rarely satisfies, so they are OR-ed."""
        async with self._connections.connect() as conn:
            rows = await conn.fetch(
                """WITH q AS (
                    SELECT nullif(replace(plainto_tsquery('english', $1)::text, '&', '|'), '')
                        ::tsquery AS query
                )
                SELECT source, section, text, ts_rank_cd(search, q.query) AS rank
                FROM guideline_chunks, q
                WHERE q.query IS NOT NULL AND search @@ q.query
                ORDER BY rank DESC, id LIMIT $2""",
                text,
                top_k,
            )
        return [_match(row, -row["rank"]) for row in rows]

    async def document(self, source: str) -> DocumentVersion | None:
        async with self._connections.connect() as conn:
            row = await conn.fetchrow("SELECT * FROM guideline_documents WHERE source = $1", source)
        return DocumentVersion(**dict(row)) if row else None

    async def replace_document(
        self,
        source: str,
        *,
        content_hash: str,
        embedding_model: str,
        chunker_version: str,
        chunks: list[StoredChunk],
    ) -> None:
        """Makes `chunks` the document's content, in one transaction: chunks
        whose id (source, position, text) is unchanged are kept as they are,
        the rest replaced. Readers see the old version or the new, never a mix
        or nothing."""
        vectors = [_vector(chunk.embedding) for chunk in chunks]
        async with self._connections.connect() as conn, conn.transaction():
            await conn.execute(
                """INSERT INTO guideline_documents
                (source, content_hash, embedding_model, chunker_version, chunk_count, ingested_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source) DO UPDATE SET content_hash = $2, embedding_model = $3,
                    chunker_version = $4, chunk_count = $5, ingested_at = $6""",
                source,
                content_hash,
                embedding_model,
                chunker_version,
                len(chunks),
                datetime.now(UTC),
            )
            await conn.execute(
                "DELETE FROM guideline_chunks WHERE source = $1 AND NOT (id = ANY($2::text[]))",
                source,
                [chunk.id for chunk in chunks],
            )
            await conn.executemany(
                """INSERT INTO guideline_chunks (id, source, position, section, text, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector) ON CONFLICT (id) DO NOTHING""",
                [
                    (chunk.id, source, chunk.position, chunk.section, chunk.text, vector)
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )

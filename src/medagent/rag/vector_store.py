"""ChromaDB adapter for guideline chunk storage and similarity search."""

import asyncio
from typing import Any

import chromadb
from chromadb.api import ClientAPI


class VectorStore:
    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str = "guidelines",
        *,
        client: ClientAPI | None = None,
    ) -> None:
        # `client`: an in-process chromadb client (EphemeralClient) for tests and
        # the retrieval eval; production talks to the ChromaDB server.
        self._client = client or chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(collection_name)

    async def ping(self) -> bool:
        """Readiness probe: does the ChromaDB server answer?"""
        await asyncio.to_thread(self._client.heartbeat)
        return True

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        # chromadb's stubs want numpy arrays / stricter mapping types than
        # its actual runtime API, which accepts these plain lists/dicts fine
        # (this is exactly what sentence-transformers produces).
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=documents,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )

    async def delete_source(self, source: str, *, except_ingestion: str) -> None:
        """Removes a document's chunks from earlier ingestions, keeping the
        ones `except_ingestion` just added -- so re-ingesting replaces a
        document instead of duplicating it, and never leaves it missing."""
        await asyncio.to_thread(
            self._collection.delete,
            # chromadb's Where stubs reject this valid nested $and/$ne filter.
            where={"$and": [{"source": source}, {"ingestion_id": {"$ne": except_ingestion}}]},  # type: ignore[dict-item]
        )

    async def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=top_k,
        )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
        ]

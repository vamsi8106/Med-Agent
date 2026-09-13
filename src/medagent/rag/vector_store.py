"""ChromaDB adapter for guideline chunk storage and similarity search."""

import asyncio
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "guidelines") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    async def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def query(self, query_embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
        ]

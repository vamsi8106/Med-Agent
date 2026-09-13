"""Local sentence-transformers embedding model (all-MiniLM-L6-v2)."""

import asyncio

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

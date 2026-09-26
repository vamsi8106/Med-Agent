"""Local sentence-transformers embedding model (all-MiniLM-L6-v2)."""

import asyncio
import threading

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)
        # encode() runs in worker threads on one shared model, and HF fast
        # tokenizers can raise "Already borrowed" if two threads use one at the
        # same time. Encodes take milliseconds, so serialize them.
        self._encode_lock = threading.Lock()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        with self._encode_lock:
            return self._model.encode(texts, convert_to_numpy=True).tolist()

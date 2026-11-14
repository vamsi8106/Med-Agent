"""Local sentence-transformers embedding model (all-MiniLM-L6-v2)."""

import asyncio
import threading

from sentence_transformers import SentenceTransformer

from medagent.core.exceptions import MedAgentError

_MODEL_NAME = "all-MiniLM-L6-v2"
# [CLS] and [SEP], added to every encoded text.
_SPECIAL_TOKENS = 2


class EmbeddingModel:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        # encode() runs in worker threads on one shared model, and HF fast
        # tokenizers can raise "Already borrowed" if two threads use one at the
        # same time. Encodes take milliseconds, so serialize them.
        self._encode_lock = threading.Lock()

    @property
    def max_tokens(self) -> int:
        """Tokens of text the model reads per input; anything beyond is
        silently cut off before embedding."""
        limit = self._model.max_seq_length
        if limit is None:
            raise MedAgentError(f"Embedding model {self.model_name} reports no max_seq_length")
        return limit - _SPECIAL_TOKENS

    def count_tokens(self, words: list[str]) -> list[int]:
        """Token count of each word, in one batched tokenizer call (for sizing
        chunks to max_tokens)."""
        if not words:
            return []
        with self._encode_lock:
            encoded = self._model.tokenizer(words, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        with self._encode_lock:
            return self._model.encode(texts, convert_to_numpy=True).tolist()

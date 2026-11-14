"""Cross-encoder re-ranker: the second stage of two-stage retrieval.

The vector index (first stage) compares one embedding of the question with one
embedding of each chunk, computed separately -- fast, but coarse: it finds the
right section and then often ranks a nearby chunk above the recommendation
that answers the question. A cross-encoder reads the question and a passage
together and scores how well the passage answers it. Too slow to run over the
whole corpus, cheap enough for ~20 candidates. `make rag-eval` measures the gain
and the latency.
"""

import asyncio
import threading

from sentence_transformers import CrossEncoder

from medagent.core.models import ClinicalEvidence


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)
        # Shared by every request; HF tokenizers are not safe under concurrent
        # use from several threads (see EmbeddingModel).
        self._lock = threading.Lock()

    async def rerank(
        self, query: str, hits: list[ClinicalEvidence], top_n: int
    ) -> list[ClinicalEvidence]:
        """The `top_n` hits that best answer `query`, best first."""
        if not hits:
            return []
        scores = await asyncio.to_thread(self._scores, query, [hit.summary for hit in hits])
        ranked = sorted(zip(scores, range(len(hits)), strict=True), reverse=True)
        return [hits[index] for _, index in ranked[:top_n]]

    def _scores(self, query: str, passages: list[str]) -> list[float]:
        # One batched call; passages longer than the model's limit are truncated
        # by the tokenizer, which is fine for chunks sized to the embedding model.
        with self._lock:
            return [float(s) for s in self._model.predict([(query, p) for p in passages])]

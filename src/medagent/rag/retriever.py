"""Guideline retriever tool: RAG lookup over ingested clinical guidelines.

Lives in rag/, not tools/custom/, because it depends on rag/embeddings and
rag/vector_store — and tools/ may only depend on core/ and infra/ per the
dependency rules in AGENTS.md. It still implements core.interfaces.BaseTool so
agents/ (which is allowed to depend on both tools/ and rag/) can register it in
a ToolRegistry alongside ordinary tools.
"""

from typing import Any

from medagent.core.interfaces import BaseTool
from medagent.core.models import ClinicalEvidence
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.reranker import CrossEncoderReranker
from medagent.rag.vector_store import VectorStore


class GuidelineRetrieverTool(BaseTool):
    name = "guideline_retriever"
    description = "Retrieves the most relevant ingested clinical guideline passages for a query."

    def __init__(
        self,
        embeddings: EmbeddingModel,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker | None = None,
        candidates: int = 20,
        hybrid: bool = False,
    ) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store
        # With a reranker or hybrid: fetch `candidates` per search, return top_k.
        self._reranker = reranker
        self._candidates = candidates
        # Hybrid: keyword (Postgres full-text) results fused with vector results.
        self._hybrid = hybrid

    # Narrowed from BaseTool.run(**kwargs: Any) to this tool's real, typed
    # params -- an intentional, safe narrowing, not a real LSP mismatch.
    async def run(  # type: ignore[override]
        self, query: str, top_k: int = 5
    ) -> list[ClinicalEvidence]:
        [query_embedding] = await self._embeddings.embed([query])
        widen = self._reranker is not None or self._hybrid
        fetch = max(top_k, self._candidates) if widen else top_k
        evidence = _evidence(await self._vector_store.query(query_embedding, top_k=fetch))
        if self._hybrid:
            keyword = _evidence(await self._vector_store.keyword_query(query, top_k=fetch))
            evidence = reciprocal_rank_fusion([evidence, keyword])[:fetch]
        if self._reranker is None:
            return evidence[:top_k]
        return await self._reranker.rerank(query, evidence, top_k)


def _evidence(matches: list[dict[str, Any]]) -> list[ClinicalEvidence]:
    return [
        ClinicalEvidence(
            title=match["metadata"].get("section") or match["metadata"].get("source", ""),
            summary=match["document"],
            source=match["metadata"].get("source"),
        )
        for match in matches
    ]


RRF_K = 60  # the usual reciprocal rank fusion constant


def reciprocal_rank_fusion(rankings: list[list[ClinicalEvidence]]) -> list[ClinicalEvidence]:
    """Merges ranked lists by summed 1 / (RRF_K + rank): a passage high in either
    list, or present in both, comes first. Needs no calibration between the
    lists' differently scaled scores (cosine distance vs ts_rank)."""
    scores: dict[str, float] = {}
    first_seen: dict[str, ClinicalEvidence] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.summary] = scores.get(hit.summary, 0.0) + 1 / (RRF_K + rank)
            first_seen.setdefault(hit.summary, hit)
    return [first_seen[text] for text in sorted(scores, key=lambda t: scores[t], reverse=True)]

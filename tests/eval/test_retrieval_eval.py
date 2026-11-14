"""Guards retrieval quality with the production chunking (make rag-eval shows all
configurations). Floors sit just under the values measured on 2026-11-14 -- raise
them when retrieval improves, never lower them to make a change pass."""

import pytest

from tests.eval.retrieval_eval import CONFIGS, GENERAL_RERANKER, available, evaluate

pytestmark = pytest.mark.skipif(
    not available(), reason="guideline PDFs not present in guidelines_pdf/ (gitignored)"
)

_PRODUCTION = [config for config in CONFIGS if config.production]


async def test_production_chunking_retrieves_and_delivers_the_answer() -> None:
    [report] = await evaluate(_PRODUCTION)

    assert report.beyond_limit == 0, "chunks exceed the embedding model's token limit"
    assert report.recall(5) >= 0.66, report.misses  # measured 0.71
    assert report.reaches_model >= 0.52  # measured 0.57


async def test_reranking_lifts_what_reaches_the_model() -> None:
    """The optional RAG_RERANKER_MODEL path (off by default for CPU latency)."""
    [reranked] = await evaluate(
        [c for c in CONFIGS if c.reranker == GENERAL_RERANKER and not c.hybrid]
    )

    assert reranked.reaches_model >= 0.71  # measured 0.76; vector only 0.57

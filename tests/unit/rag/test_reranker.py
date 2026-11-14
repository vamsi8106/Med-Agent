import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

from medagent.core.models import ClinicalEvidence
from medagent.rag.reranker import CrossEncoderReranker


def _hits(*texts: str) -> list[ClinicalEvidence]:
    return [ClinicalEvidence(title=t, summary=t, source="kdigo") for t in texts]


def _reranker(predict: object) -> CrossEncoderReranker:
    fake_model = MagicMock()
    fake_model.predict.side_effect = predict
    with patch("medagent.rag.reranker.CrossEncoder", return_value=fake_model):
        return CrossEncoderReranker("test-cross-encoder")


async def test_orders_by_score_and_keeps_top_n_in_one_batched_call() -> None:
    calls: list[list[tuple[str, str]]] = []

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        calls.append(pairs)
        return [{"weak": 0.1, "best": 0.9, "good": 0.5}[p] for _, p in pairs]

    reranker = _reranker(predict)

    ranked = await reranker.rerank("bp target", _hits("weak", "best", "good"), top_n=2)

    assert [h.summary for h in ranked] == ["best", "good"]
    assert calls == [[("bp target", "weak"), ("bp target", "best"), ("bp target", "good")]]


async def test_no_hits_means_no_model_call() -> None:
    predict = MagicMock()
    reranker = _reranker(predict)

    assert await reranker.rerank("q", [], top_n=3) == []
    predict.assert_not_called()


async def test_concurrent_reranks_never_overlap_inside_the_model() -> None:
    inside = max_inside = 0
    guard = threading.Lock()

    def slow_predict(pairs: list[tuple[str, str]]) -> list[float]:
        nonlocal inside, max_inside
        with guard:
            inside += 1
            max_inside = max(max_inside, inside)
        time.sleep(0.05)
        with guard:
            inside -= 1
        return [0.0] * len(pairs)

    reranker = _reranker(slow_predict)
    await asyncio.gather(*(reranker.rerank(f"q{i}", _hits("a", "b"), 1) for i in range(4)))

    assert max_inside == 1

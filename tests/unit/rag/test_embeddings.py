from unittest.mock import MagicMock, patch

from medagent.rag.embeddings import EmbeddingModel


async def test_embed_returns_list_of_vectors() -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = MagicMock(tolist=lambda: [[0.1, 0.2], [0.3, 0.4]])

    with patch("medagent.rag.embeddings.SentenceTransformer", return_value=fake_model):
        model = EmbeddingModel()
        vectors = await model.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    fake_model.encode.assert_called_once_with(["hello", "world"], convert_to_numpy=True)


async def test_concurrent_embeds_never_overlap_inside_the_model() -> None:
    """The shared SentenceTransformer is called from worker threads; HF fast
    tokenizers can raise "Already borrowed" if two threads use one at once, so
    encodes must be serialized."""
    import asyncio
    import threading
    import time

    inside = 0
    max_inside = 0
    guard = threading.Lock()

    def slow_encode(texts: list[str], convert_to_numpy: bool = True) -> MagicMock:
        nonlocal inside, max_inside
        with guard:
            inside += 1
            max_inside = max(max_inside, inside)
        time.sleep(0.05)
        with guard:
            inside -= 1
        return MagicMock(tolist=lambda: [[0.0] for _ in texts])

    fake_model = MagicMock()
    fake_model.encode.side_effect = slow_encode

    with patch("medagent.rag.embeddings.SentenceTransformer", return_value=fake_model):
        model = EmbeddingModel()
        await asyncio.gather(*(model.embed([f"q{i}"]) for i in range(4)))

    assert max_inside == 1

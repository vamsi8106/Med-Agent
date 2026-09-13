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

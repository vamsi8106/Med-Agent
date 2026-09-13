"""Guideline retriever tool: RAG lookup over ingested clinical guidelines.

Lives in rag/, not tools/custom/, because it depends on rag/embeddings and
rag/vector_store — and tools/ may only depend on core/ and infra/ per the
dependency rules in AGENTS.md. It still implements core.interfaces.BaseTool so
agents/ (which is allowed to depend on both tools/ and rag/) can register it in
a ToolRegistry alongside ordinary tools.
"""

from medagent.core.interfaces import BaseTool
from medagent.core.models import ClinicalEvidence
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.vector_store import VectorStore


class GuidelineRetrieverTool(BaseTool):
    name = "guideline_retriever"
    description = "Retrieves the most relevant ingested clinical guideline passages for a query."

    def __init__(self, embeddings: EmbeddingModel, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    async def run(self, query: str, top_k: int = 5) -> list[ClinicalEvidence]:
        [query_embedding] = await self._embeddings.embed([query])
        matches = await self._vector_store.query(query_embedding, top_k=top_k)

        return [
            ClinicalEvidence(
                title=match["metadata"].get("section") or match["metadata"].get("source", ""),
                summary=match["document"],
                source=match["metadata"].get("source"),
            )
            for match in matches
        ]

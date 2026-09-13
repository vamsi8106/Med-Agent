"""Evidence Agent: combines PubMed literature search with RAG over ingested guidelines."""

import asyncio

from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import AgentResult, ClinicalEvidence, Message, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger
from medagent.memory.session import SessionMemory
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.tools.mcp.medical import MedicalMCPClient

logger = get_logger(__name__)


def _extract_text(mcp_content: object) -> str:
    if isinstance(mcp_content, list):
        return " ".join(getattr(block, "text", str(block)) for block in mcp_content)
    return str(mcp_content)


class EvidenceAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLMProvider,
        medical_client: MedicalMCPClient,
        guideline_retriever: GuidelineRetrieverTool,
        session: SessionMemory | None = None,
    ) -> None:
        self._llm = llm
        self._medical_client = medical_client
        self._guideline_retriever = guideline_retriever
        self._session = session or SessionMemory()

    async def run(self, context: PatientContext, message: str) -> str:
        result = await self.gather_evidence(context, message)
        return result.summary

    async def gather_evidence(self, context: PatientContext, message: str) -> AgentResult:
        self._session.add_message(Message(role="user", content=message))

        guideline_hits, literature_content = await asyncio.gather(
            self._guideline_retriever.run(query=message, top_k=3),
            self._search_literature(message),
        )

        literature_evidence = [
            ClinicalEvidence(title=message, summary=literature_content, source="PubMed")
        ]
        evidence = [*guideline_hits, *literature_evidence]

        citations = "\n".join(f"- {e.title} (source: {e.source or 'unknown'})" for e in evidence)
        synthesis_prompt = (
            f"Summarize the clinical evidence for: {message}\n\nEvidence found:\n{citations}"
        )
        response = await self._llm.complete(
            [
                Message(
                    role="system",
                    content="You are a clinical evidence assistant. Be concise and cite sources.",
                ),
                Message(role="user", content=synthesis_prompt),
            ]
        )

        summary = f"{response.content}\n\nCitations:\n{citations}"
        self._session.add_message(Message(role="assistant", content=summary))
        return AgentResult(role=AgentRole.EVIDENCE, summary=summary, evidence=evidence)

    async def _search_literature(self, query: str) -> str:
        async with self._medical_client as client:
            content = await client.search_medical_literature(query)
        return _extract_text(content)

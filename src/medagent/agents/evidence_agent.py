"""Evidence Agent: combines PubMed literature search with RAG over ingested guidelines."""

import asyncio

from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import AgentResult, ClinicalEvidence, Message, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.tools.mcp.medical import MedicalMCPClient

logger = get_logger(__name__)


def _extract_text(mcp_content: object) -> str:
    if isinstance(mcp_content, list):
        return " ".join(
            block.get("text", str(block)) if isinstance(block, dict) else str(block)
            for block in mcp_content
        )
    return str(mcp_content)


def _format_visit_history(context: PatientContext) -> str | None:
    if not context.visits:
        return None
    # context.visits is newest-first (see PatientStore.get_patient); present
    # oldest-first so the narrative reads chronologically.
    lines = [
        f"- {visit.visit_date.date()}: complaint={visit.chief_complaint!r}, "
        f"assessment={visit.assessment!r}"
        for visit in reversed(context.visits)
    ]
    return "Recent visit history (oldest first):\n" + "\n".join(lines)


def _patient_context_preamble(context: PatientContext) -> str | None:
    visit_history = _format_visit_history(context)
    if not context.conditions and not context.allergies and not visit_history:
        return None
    parts = []
    if context.conditions:
        parts.append(f"conditions={context.conditions}")
    if context.allergies:
        parts.append(f"known allergies={context.allergies}")
    preamble = (
        f"Patient context: {'; '.join(parts)}. Tailor the evidence to this patient "
        "and flag any relevant contraindications."
        if parts
        else ""
    )
    return "\n".join(part for part in [preamble, visit_history] if part)


class EvidenceAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLMProvider,
        medical_client: MedicalMCPClient,
        guideline_retriever: GuidelineRetrieverTool,
    ) -> None:
        self._llm = llm
        self._medical_client = medical_client
        self._guideline_retriever = guideline_retriever

    async def run(self, context: PatientContext, message: str) -> str:
        result = await self.gather_evidence(context, message)
        return result.summary

    async def gather_evidence(self, context: PatientContext, message: str) -> AgentResult:
        guideline_hits, literature_content = await asyncio.gather(
            self._guideline_retriever.run(query=message, top_k=3),
            self._search_literature(message),
        )

        literature_evidence = [
            ClinicalEvidence(title=message, summary=literature_content, source="PubMed")
        ]
        evidence = [*guideline_hits, *literature_evidence]

        citations = "\n".join(f"- {e.title} (source: {e.source or 'unknown'})" for e in evidence)
        preamble = _patient_context_preamble(context)
        synthesis_prompt = "\n".join(
            [
                *([preamble] if preamble else []),
                f"Summarize the clinical evidence for: {message}",
                "",
                f"Evidence found:\n{citations}",
            ]
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
        return AgentResult(role=AgentRole.EVIDENCE, summary=summary, evidence=evidence)

    async def _search_literature(self, query: str) -> str:
        async with self._medical_client as client:
            content = await client.search_medical_literature(query)
        return _extract_text(content)

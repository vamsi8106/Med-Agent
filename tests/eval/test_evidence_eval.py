"""Live golden-dataset eval for EvidenceAgent: real Groq LLM + real medical-mcp.

Not part of the unit suite: needs a live GROQ_API_KEY and a reachable
medical-mcp-bridge. Run deliberately via `make eval`, never as part of
`make pre-commit`/CI. guideline_retriever is faked (empty results) --
RAG/ChromaDB retrieval quality is a separate concern already covered by
tests/integration/test_followup_recall.py, kept out of this eval's
dependency footprint on purpose.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.core.config import get_settings
from medagent.core.exceptions import MCPError
from medagent.core.interfaces import BaseLLMProvider
from medagent.core.models import LLMResponse, Message
from medagent.llm.registry import ProviderRegistry
from medagent.tools.mcp.medical import MedicalMCPClient
from tests.eval.evidence_golden_dataset import CASES, GoldenEvidenceCase

pytestmark = pytest.mark.skipif(
    not get_settings().groq_api_key, reason="GROQ_API_KEY not set; skipping live eval"
)


@pytest.fixture
async def evidence_agent() -> EvidenceAgent:
    settings = get_settings()
    llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
    medical_client = MedicalMCPClient(settings)
    try:
        async with medical_client as client:
            await client.search_drugs("aspirin")
    except MCPError as exc:
        pytest.skip(f"medical-mcp-bridge unreachable, skipping live eval: {exc}")

    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    return EvidenceAgent(
        llm=llm, medical_client=medical_client, guideline_retriever=guideline_retriever
    )


@pytest.mark.parametrize("case", CASES, ids=[c.patient.id for c in CASES])
async def test_evidence_response_uses_patient_context(
    evidence_agent: EvidenceAgent, case: GoldenEvidenceCase
) -> None:
    result = await evidence_agent.gather_evidence(case.patient, case.message)

    summary_lower = result.summary.lower()
    assert any(term.lower() in summary_lower for term in case.must_reference), (
        f"Response never referenced any of {case.must_reference} ({case.notes})\n"
        f"Response was:\n{result.summary}"
    )


class _CallRecorder(BaseLLMProvider):
    """Wraps the real LLM and records what each call looked like."""

    def __init__(self, inner: BaseLLMProvider) -> None:
        self._inner = inner
        self.histories: list[list[Message]] = []

    async def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        self.histories.append(list(messages))
        return await self._inner.complete(messages, tools)


async def test_real_model_takes_the_react_path_not_the_fallback(
    evidence_agent: EvidenceAgent,
) -> None:
    """EvidenceAgent falls back to the fixed pipeline on any ReAct failure, so
    the quality assertions above would still pass if tool calling silently
    broke (provider parsing, message protocol, model behavior). This pins the
    ReAct path itself: the real model must emit tool calls, the loop must feed
    their results back, and the answer must come after them."""
    recorder = _CallRecorder(evidence_agent._llm)
    evidence_agent._llm = recorder
    case = CASES[0]

    result = await evidence_agent.gather_evidence(case.patient, case.message)

    tool_result_seen = any(m.role == "tool" for history in recorder.histories for m in history)
    assert tool_result_seen, (
        "No tool result ever reached the model -- the ReAct path did not run and the "
        "deterministic fallback served this request instead "
        f"({len(recorder.histories)} LLM call(s))"
    )
    assert len(recorder.histories) >= 2
    assert result.evidence, "ReAct path retrieved no evidence"

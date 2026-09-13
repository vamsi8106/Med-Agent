"""Trial Finder Agent: searches ClinicalTrials.gov via MCP, filtered by condition."""

from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import AgentResult, ClinicalEvidence, Message, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger
from medagent.tools.mcp.healthcare import HealthcareMCPClient

logger = get_logger(__name__)


def _extract_text(mcp_content: object) -> str:
    if isinstance(mcp_content, list):
        return " ".join(getattr(block, "text", str(block)) for block in mcp_content)
    return str(mcp_content)


def _resolve_condition(context: PatientContext, message: str) -> str:
    if context.conditions:
        return context.conditions[0]
    return message


class TrialFinderAgent(BaseAgent):
    def __init__(self, llm: BaseLLMProvider, healthcare_client: HealthcareMCPClient) -> None:
        self._llm = llm
        self._healthcare_client = healthcare_client

    async def run(self, context: PatientContext, message: str) -> str:
        result = await self.find_trials(context, message)
        return result.summary

    async def find_trials(
        self, context: PatientContext, message: str, phase: str | None = None
    ) -> AgentResult:
        condition = _resolve_condition(context, message)
        async with self._healthcare_client as client:
            content = await client.clinical_trials_search(condition, phase=phase)
        trials_text = _extract_text(content)

        evidence = [
            ClinicalEvidence(
                title=f"Clinical trials for {condition}",
                summary=trials_text,
                source="ClinicalTrials.gov",
            )
        ]

        response = await self._llm.complete(
            [
                Message(
                    role="system",
                    content="You summarize clinical trial search results for a doctor.",
                ),
                Message(
                    role="user",
                    content=f"Summarize these trials for condition '{condition}':\n{trials_text}",
                ),
            ]
        )

        summary = f"{response.content}\n\nCitations:\n- {condition} (source: ClinicalTrials.gov)"
        return AgentResult(role=AgentRole.TRIAL_FINDER, summary=summary, evidence=evidence)

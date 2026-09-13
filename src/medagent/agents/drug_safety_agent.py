"""Drug Safety Agent: checks pairwise interactions and cites sources for doctors."""

import re

from medagent.core.exceptions import ToolError
from medagent.core.interfaces import BaseAgent, BaseLLMProvider
from medagent.core.models import AgentResult, DrugInteraction, Medication, Message, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger
from medagent.tools.custom.interaction_checker import InteractionCheckerTool

logger = get_logger(__name__)

_SPLIT_PATTERN = re.compile(r"\s*(?:\+|,|\band\b)\s*", flags=re.IGNORECASE)
_FOR_PATTERN = re.compile(r"\bfor\s+(.*)", flags=re.IGNORECASE)


def _extract_drug_names(message: str, context: PatientContext) -> list[str]:
    match = _FOR_PATTERN.search(message)
    segment = match.group(1) if match else message
    mentioned = [part.strip().rstrip("?.!") for part in _SPLIT_PATTERN.split(segment)]
    mentioned = [name for name in mentioned if name]

    known = [med.name for med in context.medications]
    names: list[str] = []
    for name in [*known, *mentioned]:
        if name not in names:
            names.append(name)
    return names


def _format_citations(interactions: list[DrugInteraction]) -> str:
    if not interactions:
        return "No interactions found for the listed medications."
    lines = [
        f"- {i.drug_a} + {i.drug_b}: {i.severity.value} "
        f"(source: {i.source or 'unknown'}) - {i.description}"
        for i in interactions
    ]
    return "\n".join(lines)


class DrugSafetyAgent(BaseAgent):
    def __init__(self, llm: BaseLLMProvider, interaction_checker: InteractionCheckerTool) -> None:
        self._llm = llm
        self._interaction_checker = interaction_checker

    async def run(self, context: PatientContext, message: str) -> str:
        final_answer, _ = await self._check_and_synthesize(context, message)
        return final_answer

    async def run_result(self, context: PatientContext, message: str) -> AgentResult:
        final_answer, interactions = await self._check_and_synthesize(context, message)
        return AgentResult(
            role=AgentRole.DRUG_SAFETY, summary=final_answer, interactions=interactions
        )

    async def _check_and_synthesize(
        self, context: PatientContext, message: str
    ) -> tuple[str, list[DrugInteraction]]:
        drug_names = _extract_drug_names(message, context)
        if len(drug_names) < 2:
            raise ToolError(
                f"At least two medications are required to check interactions; found: {drug_names}"
            )
        medications = [Medication(name=name) for name in drug_names]

        result = await self._interaction_checker.run(medications)
        if not result.success:
            raise ToolError(f"Interaction check failed: {result.error}")

        interactions: list[DrugInteraction] = result.data
        citations = _format_citations(interactions)

        synthesis_prompt = (
            "Summarize the following drug interaction findings for a doctor, "
            "in plain clinical language:\n" + citations
        )
        response = await self._llm.complete(
            [
                Message(
                    role="system",
                    content=(
                        "You are a clinical drug-safety assistant. Be concise and cite sources."
                    ),
                ),
                Message(role="user", content=synthesis_prompt),
            ]
        )

        final_answer = f"{response.content}\n\nCitations:\n{citations}"
        return final_answer, interactions

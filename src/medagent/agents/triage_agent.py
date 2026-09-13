"""Triage Agent: routes a doctor's message to the relevant specialist agent(s)."""

import re

from medagent.core.interfaces import BaseAgent
from medagent.core.models import PatientContext
from medagent.core.types import AgentRole

_ROUTE_KEYWORDS: dict[AgentRole, tuple[str, ...]] = {
    AgentRole.DRUG_SAFETY: ("interaction", "drug", "medication", "dose", "add "),
    AgentRole.EVIDENCE: ("evidence", "guideline", "literature", "treatment", "recommend"),
    AgentRole.TRIAL_FINDER: ("trial", "study", "studies", "research study"),
}


class TriageAgent(BaseAgent):
    async def run(self, context: PatientContext, message: str) -> str:
        roles = await self.route(context, message)
        return ", ".join(role.value for role in roles)

    async def route(self, context: PatientContext, message: str) -> list[AgentRole]:
        lowered = message.lower()
        roles = [
            role
            for role, keywords in _ROUTE_KEYWORDS.items()
            if any(re.search(re.escape(keyword), lowered) for keyword in keywords)
        ]
        return roles or [AgentRole.EVIDENCE]

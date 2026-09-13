"""New-patient workflow: triage -> parallel(drug_safety, evidence) -> report.

Depends only on core/ and agents/ per the dependency rules; persistence is
injected as core.interfaces.BaseMemory rather than importing memory/ directly.
"""

import asyncio

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.interfaces import BaseMemory
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger

logger = get_logger(__name__)


async def run_patient_assessment(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    report: ReportAgent,
    memory: BaseMemory,
    context: PatientContext,
    message: str,
) -> str:
    roles = await triage.route(context, message)

    tasks = []
    if AgentRole.DRUG_SAFETY in roles and len(context.medications) >= 2:
        tasks.append(drug_safety.run_result(context, message))
    if AgentRole.EVIDENCE in roles:
        tasks.append(evidence.gather_evidence(context, message))

    results: list[AgentResult] = list(await asyncio.gather(*tasks)) if tasks else []

    report_markdown = await report.generate(context, results)
    await memory.save_patient(context)
    return report_markdown

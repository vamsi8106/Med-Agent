"""New-patient workflow: triage -> parallel(drug_safety, evidence) -> report.

Depends only on core/ and agents/ per the dependency rules. Does NOT persist
the patient record itself: generation and persistence are deliberately
separated so a human-in-the-loop checkpoint can sit between them (see
app.py's WebSocket chat handler, which holds the report for doctor
approval/edit/rejection before saving). Callers that want unconditional
auto-save (e.g. the plain REST endpoints) save explicitly after this returns.
"""

import asyncio

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger

logger = get_logger(__name__)


async def run_patient_assessment(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    report: ReportAgent,
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

    return await report.generate(context, results)

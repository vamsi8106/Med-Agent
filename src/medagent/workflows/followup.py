"""Follow-up workflow: recall patient -> re-run relevant agents -> update record."""

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import MedAgentError
from medagent.core.interfaces import BaseMemory
from medagent.workflows.patient_assessment import run_patient_assessment


async def run_followup(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    report: ReportAgent,
    memory: BaseMemory,
    patient_id: str,
    message: str,
) -> str:
    context = await memory.get_patient(patient_id)
    if context is None:
        raise MedAgentError(f"No existing patient record for id: {patient_id}")

    return await run_patient_assessment(
        triage, drug_safety, evidence, report, memory, context, message
    )

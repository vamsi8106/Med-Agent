"""Follow-up workflow: recall patient -> re-run relevant agents -> draft report.

Returns the report alongside the PatientContext it was generated from, rather
than saving unconditionally, so a caller can gate persistence on human
approval (see app.py's WebSocket chat handler) or save immediately (REST).
"""

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import MedAgentError
from medagent.core.interfaces import BaseMemory
from medagent.core.models import PatientContext
from medagent.workflows.patient_assessment import run_patient_assessment


async def run_followup(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    report: ReportAgent,
    memory: BaseMemory,
    patient_id: str,
    message: str,
) -> tuple[str, PatientContext]:
    context = await memory.get_patient(patient_id)
    if context is None:
        raise MedAgentError(f"No existing patient record for id: {patient_id}")

    report_markdown = await run_patient_assessment(
        triage, drug_safety, evidence, report, context, message
    )
    return report_markdown, context

"""Follow-up workflow: recall patient -> re-run relevant agents -> draft report.

Returns the report alongside the PatientContext it was generated from, rather
than saving unconditionally, so a caller can gate persistence on human
approval (see app.py's WebSocket chat handler) or save immediately (REST).
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.core.exceptions import PatientNotFoundError
from medagent.core.interfaces import BaseMemory
from medagent.core.models import PatientContext
from medagent.workflows.patient_assessment import run_patient_assessment


class FollowupState(TypedDict):
    patient_id: str
    doctor_id: str | None
    message: str
    context: PatientContext | None
    report: str


def _build_graph(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
    memory: BaseMemory,
) -> StateGraph:
    async def load_patient(state: FollowupState) -> dict[str, object]:
        context = await memory.get_patient(state["patient_id"], state["doctor_id"])
        if context is None:
            raise PatientNotFoundError(f"No existing patient record for id: {state['patient_id']}")
        return {"context": context}

    async def run_assessment(state: FollowupState) -> dict[str, object]:
        context = state["context"]
        assert context is not None
        report_markdown = await run_patient_assessment(
            triage, drug_safety, evidence, trial_finder, report, context, state["message"]
        )
        return {"report": report_markdown}

    graph = StateGraph(FollowupState)
    graph.add_node("load_patient", load_patient)
    graph.add_node("run_assessment", run_assessment)
    graph.add_edge(START, "load_patient")
    graph.add_edge("load_patient", "run_assessment")
    graph.add_edge("run_assessment", END)
    return graph


async def run_followup(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
    memory: BaseMemory,
    patient_id: str,
    message: str,
    doctor_id: str | None = None,
) -> tuple[str, PatientContext]:
    graph = _build_graph(triage, drug_safety, evidence, trial_finder, report, memory).compile()
    result = await graph.ainvoke(
        {"patient_id": patient_id, "doctor_id": doctor_id, "message": message}
    )
    return result["report"], result["context"]

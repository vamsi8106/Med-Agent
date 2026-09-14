"""New-patient workflow: triage -> parallel(drug_safety, evidence, trial_finder) -> report.

Depends only on core/ and agents/ per the dependency rules. Does NOT persist
the patient record itself: generation and persistence are deliberately
separated so a human-in-the-loop checkpoint can sit between them (see
app.py's WebSocket chat handler, which holds the report for doctor
approval/edit/rejection before saving). Callers that want unconditional
auto-save (e.g. the plain REST endpoints) save explicitly after this returns.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.logging import get_logger

logger = get_logger(__name__)


class AssessmentState(TypedDict):
    context: PatientContext
    message: str
    roles: list[AgentRole]
    results: Annotated[list[AgentResult], operator.add]
    report: str


def _build_graph(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
) -> StateGraph:
    async def triage_node(state: AssessmentState) -> dict[str, object]:
        roles = await triage.route(state["context"], state["message"])
        return {"roles": roles}

    async def drug_safety_node(state: AssessmentState) -> dict[str, object]:
        result = await drug_safety.run_result(state["context"], state["message"])
        return {"results": [result]}

    async def evidence_node(state: AssessmentState) -> dict[str, object]:
        result = await evidence.gather_evidence(state["context"], state["message"])
        return {"results": [result]}

    async def trial_finder_node(state: AssessmentState) -> dict[str, object]:
        result = await trial_finder.find_trials(state["context"], state["message"])
        return {"results": [result]}

    async def report_node(state: AssessmentState) -> dict[str, object]:
        markdown = await report.generate(state["context"], state["results"])
        return {"report": markdown}

    def route_after_triage(state: AssessmentState) -> list[str]:
        roles = state["roles"]
        context = state["context"]
        branches = []
        if AgentRole.DRUG_SAFETY in roles and len(context.medications) >= 2:
            branches.append("drug_safety_node")
        if AgentRole.EVIDENCE in roles:
            branches.append("evidence_node")
        if AgentRole.TRIAL_FINDER in roles:
            branches.append("trial_finder_node")
        return branches or ["report_node"]

    graph = StateGraph(AssessmentState)
    graph.add_node("triage_node", triage_node)
    graph.add_node("drug_safety_node", drug_safety_node)
    graph.add_node("evidence_node", evidence_node)
    graph.add_node("trial_finder_node", trial_finder_node)
    graph.add_node("report_node", report_node)

    graph.add_edge(START, "triage_node")
    graph.add_conditional_edges(
        "triage_node",
        route_after_triage,
        ["drug_safety_node", "evidence_node", "trial_finder_node", "report_node"],
    )
    graph.add_edge("drug_safety_node", "report_node")
    graph.add_edge("evidence_node", "report_node")
    graph.add_edge("trial_finder_node", "report_node")
    graph.add_edge("report_node", END)
    return graph


async def run_patient_assessment(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
    context: PatientContext,
    message: str,
) -> str:
    graph = _build_graph(triage, drug_safety, evidence, trial_finder, report).compile()
    result = await graph.ainvoke({"context": context, "message": message, "results": []})
    return result["report"]

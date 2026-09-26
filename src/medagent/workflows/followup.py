"""Follow-up workflow: recall patient -> assess (subgraph) -> draft report.

Two ways to run it, both over the same graph:

- run_followup(): straight through, no approval. Returns the report alongside
  the PatientContext it was generated from, so the caller can save it
  (the REST endpoint).
- draft_followup() / resolve_followup(): the human-in-the-loop path. The graph
  pauses at an `approval` node with LangGraph's interrupt() and its state is
  held by a checkpointer, so a drafted report survives a dropped connection
  and the doctor can resume it. Nothing is saved by the workflow itself --
  workflows may not import memory/ -- the caller persists according to the
  returned ApprovalOutcome.

The assessment is embedded as a subgraph, so progress from inside it (triage,
each specialist, the report) streams out through draft_followup's callback.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.core.exceptions import PatientNotFoundError, ToolError
from medagent.core.interfaces import BaseMemory
from medagent.core.models import PatientContext
from medagent.infra.logging import get_logger
from medagent.workflows.patient_assessment import build_assessment_graph

logger = get_logger(__name__)

_APPROVAL_NODE = "approval_node"
_PROGRESS_STEPS = {
    "triage_node": "triage",
    "drug_safety_node": "drug_safety",
    "evidence_node": "evidence",
    "trial_finder_node": "trial_finder",
    "report_node": "report",
}

ProgressCallback = Callable[[dict[str, str]], Awaitable[None]]
ApprovalKind = Literal["approved", "rejected", "edited"]


class FollowupState(TypedDict):
    patient_id: str
    doctor_id: str | None
    message: str
    context: PatientContext | None
    report: str
    decision: str


@dataclass
class DraftResult:
    report: str
    context: PatientContext
    message: str


@dataclass
class ApprovalOutcome:
    kind: ApprovalKind
    report: str | None
    context: PatientContext
    message: str


def interpret_decision(decision: str, drafted_report: str) -> tuple[ApprovalKind, str | None]:
    """Maps the doctor's reply onto (kind, report to save): "approve" saves the
    draft verbatim, "reject" discards it, and any other text is their edited
    version of the report."""
    reply = decision.strip()
    if reply.lower() == "approve":
        return "approved", drafted_report
    if reply.lower() == "reject":
        return "rejected", None
    return "edited", reply


def _build_graph(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
    memory: BaseMemory,
    with_approval: bool,
) -> StateGraph:
    async def load_patient(state: FollowupState) -> dict[str, object]:
        context = await memory.get_patient(state["patient_id"], state["doctor_id"])
        if context is None:
            raise PatientNotFoundError(f"No existing patient record for id: {state['patient_id']}")
        return {"context": context}

    async def approval_node(state: FollowupState) -> dict[str, object]:
        # interrupt() pauses the run here until the caller resumes it with the
        # doctor's reply. On resume this node re-runs from the top and
        # interrupt() returns that reply, so nothing before it may have side
        # effects -- it doesn't.
        decision = interrupt({"type": "pending_approval", "report": state["report"]})
        return {"decision": decision}

    graph = StateGraph(FollowupState)
    graph.add_node("load_patient", load_patient)
    graph.add_node(
        "assess", build_assessment_graph(triage, drug_safety, evidence, trial_finder, report)
    )
    graph.add_edge(START, "load_patient")
    graph.add_edge("load_patient", "assess")
    if with_approval:
        graph.add_node(_APPROVAL_NODE, approval_node)
        graph.add_edge("assess", _APPROVAL_NODE)
        graph.add_edge(_APPROVAL_NODE, END)
    else:
        graph.add_edge("assess", END)
    return graph


def build_followup_graph(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
    memory: BaseMemory,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """With a checkpointer the graph includes the approval step (interrupt()
    needs somewhere to keep the paused state); without one it runs straight
    through."""
    graph = _build_graph(
        triage,
        drug_safety,
        evidence,
        trial_finder,
        report,
        memory,
        with_approval=checkpointer is not None,
    )
    return graph.compile(checkpointer=checkpointer)


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
    graph = build_followup_graph(triage, drug_safety, evidence, trial_finder, report, memory)
    result = await graph.ainvoke(
        {"patient_id": patient_id, "doctor_id": doctor_id, "message": message}
    )
    return result["report"], result["context"]


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _saver(graph: CompiledStateGraph) -> BaseCheckpointSaver:
    saver = graph.checkpointer
    if not isinstance(saver, BaseCheckpointSaver):
        raise ToolError("This follow-up graph was built without a checkpointer, so it cannot pause")
    return saver


def _progress_events(update: Any) -> list[dict[str, str]]:
    """Turns one `updates` chunk into progress events for the steps we report.
    A specialist that returned results completed, one that recorded a failure
    failed, and one that returned nothing was skipped (token budget)."""
    events: list[dict[str, str]] = []
    if not isinstance(update, dict):
        return events
    for node, node_update in update.items():
        step = _PROGRESS_STEPS.get(node)
        if step is None:
            continue
        node_update = node_update or {}
        if node_update.get("failures"):
            status = "failed"
        elif node in ("triage_node", "report_node") or node_update.get("results"):
            status = "completed"
        else:
            status = "skipped"
        events.append({"type": "progress", "step": step, "status": status})
    return events


async def pending_draft(graph: CompiledStateGraph, thread_id: str) -> DraftResult | None:
    """The report awaiting the doctor's decision on this thread, if any."""
    snapshot = await graph.aget_state(_config(thread_id))
    if _APPROVAL_NODE not in snapshot.next:
        return None
    values = snapshot.values
    return DraftResult(
        report=values["report"], context=values["context"], message=values["message"]
    )


async def discard_draft(graph: CompiledStateGraph, thread_id: str) -> None:
    await _saver(graph).adelete_thread(thread_id)


async def draft_followup(
    graph: CompiledStateGraph,
    *,
    patient_id: str,
    message: str,
    doctor_id: str | None,
    thread_id: str,
    on_progress: ProgressCallback | None = None,
) -> DraftResult:
    """Runs the follow-up until it pauses for approval, streaming progress.

    Always starts from a clean thread, and cleans it up if the run fails, so a
    half-finished earlier run can never leak state into this one.
    """
    await discard_draft(graph, thread_id)
    try:
        async for _namespace, update in graph.astream(
            {"patient_id": patient_id, "doctor_id": doctor_id, "message": message},
            _config(thread_id),
            stream_mode="updates",
            subgraphs=True,
        ):
            if on_progress is not None:
                for event in _progress_events(update):
                    await on_progress(event)
        draft = await pending_draft(graph, thread_id)
    except BaseException:
        await discard_draft(graph, thread_id)
        raise
    if draft is None:
        await discard_draft(graph, thread_id)
        raise ToolError("Follow-up finished without producing a draft to approve")
    return draft


async def resolve_followup(
    graph: CompiledStateGraph, *, thread_id: str, decision: str
) -> ApprovalOutcome:
    """Resumes the paused run with the doctor's reply and clears the thread."""
    draft = await pending_draft(graph, thread_id)
    if draft is None:
        raise ToolError("No draft is awaiting approval")
    try:
        result = await graph.ainvoke(Command(resume=decision), _config(thread_id))
    finally:
        await discard_draft(graph, thread_id)
    kind, report = interpret_decision(result["decision"], draft.report)
    return ApprovalOutcome(kind=kind, report=report, context=draft.context, message=draft.message)

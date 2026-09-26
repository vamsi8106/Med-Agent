"""New-patient workflow: triage -> parallel(drug_safety, evidence, trial_finder) -> report.

Depends only on core/, infra/ and agents/ per the dependency rules. Does NOT
persist the patient record itself: generation and persistence are deliberately
separated so a human-in-the-loop checkpoint can sit between them (see
followup.py's approval node and app.py's WebSocket chat handler). Callers that
want unconditional auto-save (e.g. the plain REST endpoints) save explicitly
after this returns.

The graph is self-contained -- its first node creates the per-run token
tracker -- so it runs standalone via run_patient_assessment() and also embeds
as a subgraph node in the follow-up workflow.

A specialist that fails with a MedAgentError does not sink the run: the other
sections and the deterministic allergy/lab sections still reach the doctor,
with a note saying what was unavailable. Only if every routed specialist fails
is there nothing clinical to report, and AllAgentsFailedError is raised.
"""

import operator
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.core.config import get_settings
from medagent.core.exceptions import AllAgentsFailedError, MedAgentError, UpstreamError
from medagent.core.models import AgentFailure, AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.infra.agent_run import AgentRunTracker
from medagent.infra.logging import get_logger
from medagent.infra.metrics import agent_run_total

logger = get_logger(__name__)

_DEFAULT_FAILURE_TEXT = "the check could not be completed"


class AssessmentState(TypedDict):
    context: PatientContext
    message: str
    token_budget: int | None
    roles: list[AgentRole]
    results: Annotated[list[AgentResult], operator.add]
    failures: Annotated[list[AgentFailure], operator.add]
    report: str
    tracker: AgentRunTracker


def _doctor_safe_reason(exc: MedAgentError) -> str:
    """Fixed text, never the raw exception message: a provider error can embed
    account identifiers (the Groq 413 we hit contained the organisation id).
    A rate limit says so and how long to wait; an outage says unavailable."""
    if isinstance(exc, UpstreamError):
        return exc.public_message()
    return _DEFAULT_FAILURE_TEXT


def _role_label(role: AgentRole) -> str:
    return role.value.replace("_", " ").title()


def _completeness_note(tracker: AgentRunTracker, failures: list[AgentFailure]) -> str | None:
    lines: list[str] = []
    if failures:
        unavailable = ", ".join(f"{_role_label(f.role)} ({f.error})" for f in failures)
        lines.append(f"Unavailable: {unavailable}. Re-run to retry.")
    skipped = tracker.skipped_steps()
    if skipped:
        lines.append(
            f"Skipped due to token budget ({tracker.tokens_used}/{tracker.max_tokens} used): "
            f"{', '.join(skipped)}. Re-run individually if this evidence is needed."
        )
    return "## Note\n\n" + "\n\n".join(lines) if lines else None


async def _run_specialist(
    state: AssessmentState,
    name: str,
    role: AgentRole,
    call: Callable[[], Awaitable[AgentResult]],
) -> dict[str, object]:
    """Budget check, run, and turn a MedAgentError into a recorded failure.

    Only MedAgentError is caught: those are the expected operational failures
    (provider down, MCP unreachable, bad input). Anything else is a bug and
    must stay loud rather than be quietly reported as "unavailable".
    """
    tracker = state["tracker"]
    if tracker.over_budget():
        tracker.skip(name)
        agent_run_total.labels(agent_role=role.value, outcome="skipped").inc()
        return {}
    try:
        result = await call()
    except MedAgentError as exc:
        logger.warning(
            "specialist_failed", step=name, error_type=type(exc).__name__, detail=str(exc)
        )
        tracker.fail(name, type(exc).__name__)
        agent_run_total.labels(agent_role=role.value, outcome="failed").inc()
        return {
            "failures": [
                AgentFailure(
                    role=role,
                    error=_doctor_safe_reason(exc),
                    reason=getattr(exc, "reason", "error"),
                    retry_after=getattr(exc, "retry_after", None),
                )
            ]
        }
    tracker.record(name, result.usage)
    agent_run_total.labels(agent_role=role.value, outcome="completed").inc()
    return {"results": [result]}


def _build_graph(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
) -> StateGraph:
    async def init_run_node(state: AssessmentState) -> dict[str, object]:
        budget = state.get("token_budget") or get_settings().agent_max_tokens_per_run
        return {"tracker": AgentRunTracker(max_tokens=budget, run_id=str(uuid.uuid4()))}

    async def triage_node(state: AssessmentState) -> dict[str, object]:
        roles = await triage.route(state["context"], state["message"])
        return {"roles": roles}

    async def drug_safety_node(state: AssessmentState) -> dict[str, object]:
        return await _run_specialist(
            state,
            "drug_safety",
            AgentRole.DRUG_SAFETY,
            lambda: drug_safety.run_result(state["context"], state["message"]),
        )

    async def evidence_node(state: AssessmentState) -> dict[str, object]:
        return await _run_specialist(
            state,
            "evidence",
            AgentRole.EVIDENCE,
            lambda: evidence.gather_evidence(state["context"], state["message"]),
        )

    async def trial_finder_node(state: AssessmentState) -> dict[str, object]:
        return await _run_specialist(
            state,
            "trial_finder",
            AgentRole.TRIAL_FINDER,
            lambda: trial_finder.find_trials(state["context"], state["message"]),
        )

    async def report_node(state: AssessmentState) -> dict[str, object]:
        tracker = state["tracker"]
        results = state.get("results", [])
        failures = state.get("failures", [])
        if failures and not results:
            # Every routed specialist failed: nothing clinical to report, and
            # returning an empty "unavailable" report would let the REST
            # endpoints auto-save it as a visit that later prompts replay.
            reasons = ", ".join(f"{f.role.value} ({f.error})" for f in failures)
            # All throttled -> tell the client to back off (429 + Retry-After)
            # rather than report an outage; any other mix stays a 502.
            throttled = all(f.reason == "rate_limited" for f in failures)
            waits = [f.retry_after for f in failures if f.retry_after is not None]
            raise AllAgentsFailedError(
                f"No specialist agent could complete: {reasons}",
                reason="rate_limited" if throttled else "error",
                retry_after=max(waits) if throttled and waits else None,
            )

        markdown = await report.generate(state["context"], results)
        note = _completeness_note(tracker, failures)
        if note:
            markdown += "\n\n" + note
        logger.info("agent_run_completed", run_id=tracker.run_id, trace=tracker.trace_summary())
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
    graph.add_node("init_run_node", init_run_node)
    graph.add_node("triage_node", triage_node)
    graph.add_node("drug_safety_node", drug_safety_node)
    graph.add_node("evidence_node", evidence_node)
    graph.add_node("trial_finder_node", trial_finder_node)
    graph.add_node("report_node", report_node)

    graph.add_edge(START, "init_run_node")
    graph.add_edge("init_run_node", "triage_node")
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


def build_assessment_graph(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
) -> CompiledStateGraph:
    return _build_graph(triage, drug_safety, evidence, trial_finder, report).compile()


async def run_patient_assessment(
    triage: TriageAgent,
    drug_safety: DrugSafetyAgent,
    evidence: EvidenceAgent,
    trial_finder: TrialFinderAgent,
    report: ReportAgent,
    context: PatientContext,
    message: str,
    max_tokens: int | None = None,
) -> str:
    graph = build_assessment_graph(triage, drug_safety, evidence, trial_finder, report)
    result = await graph.ainvoke(
        {"context": context, "message": message, "token_budget": max_tokens}
    )
    return result["report"]

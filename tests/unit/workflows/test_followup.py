from unittest.mock import AsyncMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import (
    AllAgentsFailedError,
    PatientNotFoundError,
    ProviderError,
    ToolError,
)
from medagent.core.models import AgentResult, PatientContext
from medagent.core.types import AgentRole
from medagent.llm.mock_provider import MockLLMProvider
from medagent.workflows.checkpointing import build_checkpointer
from medagent.workflows.followup import (
    build_followup_graph,
    draft_followup,
    interpret_decision,
    pending_draft,
    resolve_followup,
    run_followup,
)


async def test_followup_raises_for_unknown_patient() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = None

    with pytest.raises(PatientNotFoundError):
        await run_followup(
            TriageAgent(),
            AsyncMock(),
            AsyncMock(),
            AsyncMock(),
            ReportAgent(),
            memory,
            "P-UNKNOWN",
            "hi",
        )


async def test_followup_recalls_and_runs_assessment() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-600", name="Patient Alpha", age=50, sex="F"
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = AsyncMock()
    medical_client.__aenter__.return_value = medical_client
    medical_client.search_medical_literature.return_value = "n/a"
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="Stable, continue current plan."),
        medical_client=medical_client,
        guideline_retriever=guideline_retriever,
    )

    report, patient = await run_followup(
        TriageAgent(),
        AsyncMock(),
        evidence,
        AsyncMock(),
        ReportAgent(),
        memory,
        "P-TEST-600",
        "follow-up visit",
    )

    assert "Patient Alpha" in report
    assert patient.id == "P-TEST-600"
    memory.save_patient.assert_not_awaited()


async def test_followup_forwards_doctor_id_to_memory() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-601", name="Patient Beta", age=45, sex="M", doctor_id="DR-TEST-001"
    )
    guideline_retriever = AsyncMock()
    guideline_retriever.run.return_value = []
    medical_client = AsyncMock()
    medical_client.__aenter__.return_value = medical_client
    medical_client.search_medical_literature.return_value = "n/a"
    evidence = EvidenceAgent(
        llm=MockLLMProvider(fixed_response="n/a"),
        medical_client=medical_client,
        guideline_retriever=guideline_retriever,
    )

    await run_followup(
        TriageAgent(),
        AsyncMock(),
        evidence,
        AsyncMock(),
        ReportAgent(),
        memory,
        "P-TEST-601",
        "follow-up visit",
        doctor_id="DR-TEST-001",
    )

    memory.get_patient.assert_awaited_once_with("P-TEST-601", "DR-TEST-001")


# --- approval flow (interrupt + checkpointer) ---------------------------------


class _CountingEvidence:
    """Returns a distinct summary per call so state leaking between runs shows."""

    def __init__(self) -> None:
        self.calls = 0

    async def gather_evidence(self, context: PatientContext, message: str) -> AgentResult:
        self.calls += 1
        return AgentResult(role=AgentRole.EVIDENCE, summary=f"evidence run {self.calls}")


def _approval_graph(evidence: object, trial_finder: object | None = None) -> CompiledStateGraph:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-610", name="Patient Alpha", age=50, sex="F", doctor_id="DR-1"
    )
    return build_followup_graph(
        TriageAgent(),
        AsyncMock(),
        evidence,  # type: ignore[arg-type]
        trial_finder or AsyncMock(),  # type: ignore[arg-type]
        ReportAgent(),
        memory,
        checkpointer=build_checkpointer(),
    )


async def _draft(graph: CompiledStateGraph, thread: str = "t1", message: str = "any updates?"):
    return await draft_followup(
        graph, patient_id="P-TEST-610", message=message, doctor_id="DR-1", thread_id=thread
    )


async def test_draft_pauses_for_approval_and_can_be_read_back() -> None:
    graph = _approval_graph(_CountingEvidence())

    draft = await _draft(graph)

    assert "evidence run 1" in draft.report
    assert draft.message == "any updates?"
    assert draft.context.id == "P-TEST-610"
    pending = await pending_draft(graph, "t1")
    assert pending is not None
    assert pending.report == draft.report


async def test_no_pending_draft_before_one_is_created() -> None:
    graph = _approval_graph(_CountingEvidence())
    assert await pending_draft(graph, "nobody") is None


async def test_approve_returns_the_draft_and_clears_the_thread() -> None:
    graph = _approval_graph(_CountingEvidence())
    draft = await _draft(graph)

    outcome = await resolve_followup(graph, thread_id="t1", decision="approve")

    assert outcome.kind == "approved"
    assert outcome.report == draft.report
    assert outcome.message == "any updates?"
    assert outcome.context.id == "P-TEST-610"
    assert await pending_draft(graph, "t1") is None


async def test_reject_discards_the_report() -> None:
    graph = _approval_graph(_CountingEvidence())
    await _draft(graph)

    outcome = await resolve_followup(graph, thread_id="t1", decision="reject")

    assert outcome.kind == "rejected"
    assert outcome.report is None


async def test_any_other_reply_is_the_doctors_edited_report() -> None:
    graph = _approval_graph(_CountingEvidence())
    await _draft(graph)

    outcome = await resolve_followup(graph, thread_id="t1", decision="  My own wording.  ")

    assert outcome.kind == "edited"
    assert outcome.report == "My own wording."


async def test_resolving_with_no_draft_pending_raises() -> None:
    graph = _approval_graph(_CountingEvidence())
    with pytest.raises(ToolError, match="No draft"):
        await resolve_followup(graph, thread_id="t1", decision="approve")


async def test_a_new_draft_starts_clean_and_does_not_inherit_the_previous_one() -> None:
    graph = _approval_graph(_CountingEvidence())
    await _draft(graph, message="first")

    second = await _draft(graph, message="second")

    assert "evidence run 2" in second.report
    assert "evidence run 1" not in second.report
    assert second.message == "second"


async def test_threads_are_independent() -> None:
    graph = _approval_graph(_CountingEvidence())
    await _draft(graph, thread="alice:P1", message="a")
    await _draft(graph, thread="bob:P1", message="b")

    await resolve_followup(graph, thread_id="alice:P1", decision="reject")

    assert await pending_draft(graph, "alice:P1") is None
    still_pending = await pending_draft(graph, "bob:P1")
    assert still_pending is not None
    assert still_pending.message == "b"


async def test_progress_events_report_each_step_in_order() -> None:
    graph = _approval_graph(_CountingEvidence())
    events: list[dict[str, str]] = []

    async def record(event: dict[str, str]) -> None:
        events.append(event)

    await draft_followup(
        graph,
        patient_id="P-TEST-610",
        message="any updates?",
        doctor_id="DR-1",
        thread_id="t1",
        on_progress=record,
    )

    assert [(e["step"], e["status"]) for e in events] == [
        ("triage", "completed"),
        ("evidence", "completed"),
        ("report", "completed"),
    ]
    assert all(e["type"] == "progress" for e in events)


async def test_progress_marks_a_failed_specialist_and_the_draft_is_still_produced() -> None:
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError("boom")
    trial_finder = AsyncMock()
    trial_finder.find_trials.return_value = AgentResult(
        role=AgentRole.TRIAL_FINDER, summary="one trial found"
    )
    graph = _approval_graph(evidence, trial_finder)
    events: list[dict[str, str]] = []

    async def record(event: dict[str, str]) -> None:
        events.append(event)

    draft = await draft_followup(
        graph,
        patient_id="P-TEST-610",
        message="any evidence or clinical trials?",
        doctor_id="DR-1",
        thread_id="t1",
        on_progress=record,
    )

    by_step = {e["step"]: e["status"] for e in events}
    assert by_step["evidence"] == "failed"
    assert by_step["trial_finder"] == "completed"
    assert "one trial found" in draft.report
    assert "Unavailable: Evidence" in draft.report


async def test_a_failed_run_leaves_no_thread_behind() -> None:
    evidence = AsyncMock()
    evidence.gather_evidence.side_effect = ProviderError("boom")
    graph = _approval_graph(evidence)

    with pytest.raises(AllAgentsFailedError):
        await _draft(graph)

    assert await pending_draft(graph, "t1") is None
    snapshot = await graph.aget_state({"configurable": {"thread_id": "t1"}})
    assert not snapshot.values


async def test_unknown_patient_leaves_no_thread_behind() -> None:
    memory = AsyncMock()
    memory.get_patient.return_value = None
    graph = build_followup_graph(
        TriageAgent(),
        AsyncMock(),
        AsyncMock(),
        AsyncMock(),
        ReportAgent(),
        memory,
        checkpointer=build_checkpointer(),
    )

    with pytest.raises(PatientNotFoundError):
        await _draft(graph)

    assert await pending_draft(graph, "t1") is None


async def test_a_graph_built_without_a_checkpointer_cannot_pause() -> None:
    graph = build_followup_graph(
        TriageAgent(), AsyncMock(), AsyncMock(), AsyncMock(), ReportAgent(), AsyncMock()
    )
    with pytest.raises(ToolError, match="checkpointer"):
        await _draft(graph)


@pytest.mark.parametrize(
    ("reply", "kind", "saved"),
    [
        ("approve", "approved", "the draft"),
        ("  APPROVE ", "approved", "the draft"),
        ("reject", "rejected", None),
        ("Reject", "rejected", None),
        ("approve with changes", "edited", "approve with changes"),
        ("  edited text  ", "edited", "edited text"),
    ],
)
def test_interpret_decision(reply: str, kind: str, saved: str | None) -> None:
    assert interpret_decision(reply, "the draft") == (kind, saved)

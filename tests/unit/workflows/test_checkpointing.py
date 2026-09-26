import warnings
from datetime import UTC, datetime, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from medagent.core.models import (
    AgentFailure,
    AgentResult,
    ClinicalEvidence,
    DrugInteraction,
    LabResult,
    Medication,
    PatientContext,
    Visit,
)
from medagent.core.types import AgentRole, EvidenceGrade, InteractionSeverity
from medagent.infra.agent_run import AgentRunTracker
from medagent.workflows.checkpointing import (
    build_checkpointer,
    build_serializer,
    purge_expired_drafts,
)


def _full_state() -> dict:
    now = datetime.now(UTC)
    tracker = AgentRunTracker(max_tokens=100, run_id="run-1")
    tracker.record("evidence", {"prompt_tokens": 5})
    tracker.fail("drug_safety", "ProviderError")
    return {
        "context": PatientContext(
            id="P-TEST-1",
            name="Patient Alpha",
            age=60,
            sex="F",
            medications=[Medication(name="Metformin")],
            lab_results=[
                LabResult(test_name="HbA1c", value=9.0, unit="%", collected_at=now),
            ],
            visits=[Visit(visit_date=now, assessment="x")],
        ),
        "results": [
            AgentResult(
                role=AgentRole.EVIDENCE,
                summary="s",
                evidence=[ClinicalEvidence(title="t", summary="s", grade=EvidenceGrade.A)],
                interactions=[
                    DrugInteraction(
                        drug_a="a",
                        drug_b="b",
                        severity=InteractionSeverity.MAJOR,
                        description="d",
                        checked_at=now,
                    )
                ],
            )
        ],
        "failures": [AgentFailure(role=AgentRole.DRUG_SAFETY, error="down")],
        "roles": [AgentRole.EVIDENCE],
        "tracker": tracker,
    }


def test_every_type_in_the_graph_state_round_trips_without_allowlist_warnings() -> None:
    """LangGraph warns that unregistered checkpoint types "will be blocked in a
    future version". A type added to the graph state but missing from the
    allowlist would silently break resume on that upgrade -- so fail loudly now."""
    serializer = build_serializer()
    state = _full_state()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = serializer.loads_typed(serializer.dumps_typed(state))

    assert restored["context"] == state["context"]
    assert restored["results"] == state["results"]
    assert restored["failures"] == state["failures"]
    assert restored["roles"] == state["roles"]
    assert restored["tracker"] == state["tracker"]


class _S(TypedDict):
    value: int


def _tiny_graph_with_two_threads():
    saver = build_checkpointer()
    graph = StateGraph(_S)
    graph.add_node("n", lambda s: {"value": s["value"] + 1})
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    return saver, graph.compile(checkpointer=saver)


async def test_purge_removes_only_threads_older_than_the_ttl() -> None:
    saver, graph = _tiny_graph_with_two_threads()
    await graph.ainvoke({"value": 1}, {"configurable": {"thread_id": "old"}})
    await graph.ainvoke({"value": 1}, {"configurable": {"thread_id": "also-old"}})

    removed = await purge_expired_drafts(
        saver, timedelta(minutes=30), now=datetime.now(UTC) + timedelta(hours=1)
    )

    assert removed == 2
    assert not (await graph.aget_state({"configurable": {"thread_id": "old"}})).values


async def test_purge_keeps_recent_threads() -> None:
    saver, graph = _tiny_graph_with_two_threads()
    await graph.ainvoke({"value": 1}, {"configurable": {"thread_id": "fresh"}})

    removed = await purge_expired_drafts(saver, timedelta(minutes=30))

    assert removed == 0
    assert (await graph.aget_state({"configurable": {"thread_id": "fresh"}})).values == {"value": 2}

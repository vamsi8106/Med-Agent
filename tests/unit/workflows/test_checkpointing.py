import warnings
from datetime import UTC, datetime, timedelta
from typing import TypedDict
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.exceptions import MemoryError as MedAgentMemoryError
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
from medagent.memory.persistent import PersistentStore
from medagent.memory.schema import CHECKPOINT_SAVER_MIGRATIONS
from medagent.workflows.checkpointing import (
    build_checkpointer,
    build_serializer,
    open_postgres_checkpointer,
    purge_expired_drafts,
)
from medagent.workflows.followup import (
    build_followup_graph,
    draft_followup,
    pending_draft,
    resolve_followup,
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


# --- Postgres checkpointer (STATE_BACKEND=postgres) ----------------------------


def _postgres_approval_graph(saver: BaseCheckpointSaver) -> CompiledStateGraph:
    memory = AsyncMock()
    memory.get_patient.return_value = PatientContext(
        id="P-TEST-620", name="Patient Alpha", age=50, sex="F", doctor_id="DR-1"
    )
    evidence = AsyncMock()
    evidence.gather_evidence.return_value = AgentResult(
        role=AgentRole.EVIDENCE, summary="Stable on current plan."
    )
    return build_followup_graph(
        TriageAgent(), AsyncMock(), evidence, AsyncMock(), ReportAgent(), memory, checkpointer=saver
    )


async def test_a_draft_made_on_one_replica_is_resumed_on_another(pg_dsn: str) -> None:
    saver_a, pool_a = await open_postgres_checkpointer(pg_dsn)
    saver_b, pool_b = await open_postgres_checkpointer(pg_dsn)
    try:
        graph_a, graph_b = _postgres_approval_graph(saver_a), _postgres_approval_graph(saver_b)

        draft = await draft_followup(
            graph_a,
            patient_id="P-TEST-620",
            message="any updates?",
            doctor_id="DR-1",
            thread_id="U1:P-TEST-620",
        )
        resumed = await pending_draft(graph_b, "U1:P-TEST-620")
        assert resumed is not None
        assert resumed.report == draft.report
        assert resumed.context.id == "P-TEST-620"  # allowlisted types survive the round trip

        outcome = await resolve_followup(graph_b, thread_id="U1:P-TEST-620", decision="approve")
        assert outcome.kind == "approved"
        assert outcome.report == draft.report
        assert await pending_draft(graph_a, "U1:P-TEST-620") is None
    finally:
        await pool_a.close()
        await pool_b.close()


async def test_postgres_checkpointer_refuses_an_out_of_date_schema(pg_dsn: str) -> None:
    store = PersistentStore(pg_dsn)
    async with store.connect() as conn:
        await conn.execute(
            "DELETE FROM checkpoint_migrations WHERE v = (SELECT max(v) FROM checkpoint_migrations)"
        )
    try:
        with pytest.raises(MedAgentMemoryError, match="add an Alembic migration"):
            await open_postgres_checkpointer(pg_dsn)
    finally:
        async with store.connect() as conn:
            await conn.execute(
                "INSERT INTO checkpoint_migrations (v) VALUES ($1)",
                len(CHECKPOINT_SAVER_MIGRATIONS) - 1,
            )
        await store.close()

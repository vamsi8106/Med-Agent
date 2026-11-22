"""Runs agent-eval cases end to end through the real workflows.

The agents are built by the app's own `build_specialists` (api/state.py), with
a recording LLM and spy MCP clients, over a throwaway Postgres holding a real
pgvector guideline index and real patient records. Each case runs the
workflow it names:

- assessment: triage -> specialists -> report (`build_assessment_graph`);
- follow-up: the patient's last visit is saved, then the current record, and
  the follow-up graph runs to its approval pause, then is approved
  (`draft_followup` / `resolve_followup`);
- drug-check: the current record is saved and a new drug is checked against
  it (`run_drug_check`).

What happened is observed from outside (agent_observers.py) and turned into a
`RunRecord` (agent_checks.py). Nothing in `src/` changes for the eval.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medagent.api.state import Specialists, build_specialists
from medagent.core.config import Settings, get_settings
from medagent.core.exceptions import MedAgentError
from medagent.core.models import PatientContext
from medagent.core.types import AgentRole
from medagent.llm.registry import ProviderRegistry
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.pipeline import IngestionPipeline
from medagent.rag.vector_store import VectorStore
from medagent.workflows.checkpointing import build_checkpointer
from medagent.workflows.drug_check import run_drug_check
from medagent.workflows.followup import build_followup_graph, draft_followup, resolve_followup
from medagent.workflows.patient_assessment import build_assessment_graph
from tests.eval.agent_cases import AgentCase
from tests.eval.agent_checks import Observed, RunRecord, record_run
from tests.eval.agent_judge import llm_judge
from tests.eval.agent_observers import (
    FAILING_SERVER,
    Probe,
    RecordingLLM,
    SpyHealthcareClient,
    SpyMedicalClient,
    SpyResearchClient,
    capture_spans,
    observe_specialists,
    react_counts,
)

DEMO_GUIDELINE = ("synthetic-t2dm", "docs/demo/synthetic-t2dm-guideline.txt")
EVAL_DOCTOR = "DR-EVAL-001"
EVAL_RUN_ID = f"agent-eval-{uuid.uuid4().hex[:8]}"

Judge = Callable[[AgentCase, str, list[str]], Awaitable[dict[str, float] | None]]


@dataclass
class Suite:
    llm: RecordingLLM
    probe: Probe
    spans: InMemorySpanExporter
    specialists: Specialists
    assessment: CompiledStateGraph
    followup: CompiledStateGraph
    patients: PatientStore
    guideline_sources: list[str]
    model: str
    judge: Judge | None = None


@asynccontextmanager
async def prepared_suite(
    dsn: str,
    settings: Settings | None = None,
    *,
    guideline_texts: dict[str, str] | None = None,
    with_judge: bool = False,
) -> AsyncIterator[Suite]:
    """Builds the agents as the app does, over the Postgres at `dsn` (schema
    applied), with the demo guideline plus `guideline_texts` ingested."""
    settings = settings or get_settings()
    probe = Probe()
    llm = RecordingLLM(ProviderRegistry.get_provider(settings.llm_provider, settings))
    store = PersistentStore(dsn)
    embeddings = EmbeddingModel()
    vectors = VectorStore(store)
    pipeline = IngestionPipeline(embeddings, vectors)
    texts = {DEMO_GUIDELINE[0]: _read(DEMO_GUIDELINE[1]), **(guideline_texts or {})}
    for source, text in texts.items():
        await pipeline.ingest_text(text, source)

    specialists = build_specialists(
        settings,
        llm,
        vectors,
        embeddings,
        medical=SpyMedicalClient(settings, probe),
        healthcare=SpyHealthcareClient(settings, probe),
        research=SpyResearchClient(settings, probe),
    )
    observe_specialists(specialists, probe)
    patients = PatientStore(store)
    agents = (
        specialists.triage,
        specialists.drug_safety,
        specialists.evidence,
        specialists.trial_finder,
        specialists.report,
    )
    model = (
        settings.groq_model
        if settings.llm_provider == "groq"
        else (settings.llm_model or settings.llm_provider)
    )

    judge: Judge | None = None
    if with_judge:
        inner = llm.inner

        async def judge(
            case: AgentCase, report: str, sources: list[str]
        ) -> dict[str, float] | None:
            return await llm_judge(inner, case, report, sources)

    try:
        yield Suite(
            llm=llm,
            probe=probe,
            spans=capture_spans(),
            specialists=specialists,
            assessment=build_assessment_graph(*agents),
            followup=build_followup_graph(*agents, patients, checkpointer=build_checkpointer()),
            patients=patients,
            guideline_sources=sorted(texts),
            model=model,
            judge=judge,
        )
    finally:
        await store.close()


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# --- running a case --------------------------------------------------------------------


async def run_case(
    suite: Suite,
    case: AgentCase,
    repeat: int,
    *,
    max_attempts: int = 3,
    infra_wait_seconds: float = 30.0,
    max_wait_seconds: float = 120.0,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> RunRecord:
    """One run of `case`; retried while it hits upstream trouble, then
    recorded as infra if it still does. Each retry waits at least as long as
    the LLM asked (Retry-After), and longer on each attempt."""
    record = RunRecord(case.id, case.category, repeat)
    for attempt in range(1, max_attempts + 1):
        record = await _run_once(suite, case, repeat)
        record.attempts = attempt
        if not record.infra:
            return record
        if (record.retry_after or 0.0) > max_wait_seconds:
            # A quota, not a blip (Groq asks for 10-20 min once the daily
            # token budget is spent): retrying cannot help, so stop here.
            record.quota_exhausted = True
            return record
        if attempt < max_attempts:
            await sleep(retry_wait(record, attempt, infra_wait_seconds, max_wait_seconds))
    return record


def retry_wait(record: RunRecord, attempt: int, base: float, cap: float) -> float:
    return min(cap, max(record.retry_after or 0.0, base * attempt))


@dataclass
class _Outcome:
    report: str = ""
    tracker: Any = None
    approval_ok: bool | None = None


async def _run_once(suite: Suite, case: AgentCase, repeat: int) -> RunRecord:
    failing = FAILING_SERVER[case.fail_role] if case.fail_role else None
    suite.llm.calls.clear()
    suite.probe.reset(case.injection, failing)
    suite.spans.clear()
    react_before = react_counts()
    record = RunRecord(case.id, case.category, repeat, workflow=case.workflow)
    # A fresh id per run: a follow-up's "changes since last visit" must be
    # against this run's baseline, never an earlier repeat's.
    patient_id = f"{case.patient.id}-{uuid.uuid4().hex[:6]}"

    started = time.perf_counter()
    outcome = _Outcome()
    raised_reason: str | None = None
    try:
        with _langfuse_attributes(case, repeat):
            outcome = await _WORKFLOWS[case.workflow](suite, case, patient_id)
    except MedAgentError as exc:
        record.raised = type(exc).__name__
        # UpstreamError and AllAgentsFailedError carry why: an outage or a
        # rate limit is infra, not an agent error.
        reason = getattr(exc, "reason", None)
        raised_reason = reason if isinstance(reason, str) else None
    record.seconds = round(time.perf_counter() - started, 2)

    seen = Observed(
        results=list(suite.probe.results),
        roles=suite.probe.roles if suite.probe.roles is not None else _direct_roles(case),
        report=outcome.report,
        spans=list(suite.spans.get_finished_spans()),
        calls=list(suite.llm.calls),
        react_before=react_before,
        react_after=react_counts(),
        probe=suite.probe,
        patient_id=patient_id,
        raised_reason=raised_reason,
    )
    record_run(record, case, seen, suite.guideline_sources)
    record.approval_ok = outcome.approval_ok
    if outcome.tracker is not None:
        record.budget_skipped = list(outcome.tracker.skipped_steps())
    if suite.judge is not None and outcome.report and not record.infra:
        sources = [e.summary for r in seen.results for e in r.evidence]
        record.judge = await suite.judge(case, outcome.report, sources)
    return record


def _direct_roles(case: AgentCase) -> list[AgentRole]:
    """Drug-check calls Drug Safety directly, without triage."""
    return [AgentRole.DRUG_SAFETY] if case.workflow == "drug_check" else []


# --- the three workflows ---------------------------------------------------------------


async def _save(suite: Suite, context: PatientContext, patient_id: str) -> None:
    """Saves the record as the eval doctor's; visits are saved separately,
    as snapshots (save_visit)."""
    await suite.patients.save_patient(
        context.model_copy(update={"id": patient_id, "doctor_id": EVAL_DOCTOR, "visits": []})
    )


async def _assessment(suite: Suite, case: AgentCase, patient_id: str) -> _Outcome:
    context = case.patient.model_copy(update={"id": patient_id})
    state = await suite.assessment.ainvoke(
        {"context": context, "message": case.message, "token_budget": None}
    )
    return _Outcome(report=state.get("report", ""), tracker=state.get("tracker"))


async def _followup(suite: Suite, case: AgentCase, patient_id: str) -> _Outcome:
    assert case.previous is not None
    await _save(suite, case.previous, patient_id)
    last = case.previous.visits[-1] if case.previous.visits else None
    await suite.patients.save_visit(
        patient_id,
        EVAL_DOCTOR,
        last.chief_complaint if last and last.chief_complaint else "review",
        last.assessment if last and last.assessment else "",
    )
    await _save(suite, case.patient, patient_id)

    thread = f"eval-{patient_id}"
    draft = await draft_followup(
        suite.followup,
        patient_id=patient_id,
        message=case.message,
        doctor_id=EVAL_DOCTOR,
        thread_id=thread,
    )
    approved = await resolve_followup(suite.followup, thread_id=thread, decision="approve")
    return _Outcome(
        report=draft.report,
        approval_ok=bool(draft.report)
        and approved.kind == "approved"
        and approved.report == draft.report,
    )


async def _drug_check(suite: Suite, case: AgentCase, patient_id: str) -> _Outcome:
    assert case.new_drug is not None
    await _save(suite, case.patient, patient_id)
    outcome = await run_drug_check(
        suite.specialists.drug_safety, suite.patients, patient_id, case.new_drug, EVAL_DOCTOR
    )
    return _Outcome(report=outcome.answer)


_WORKFLOWS: dict[str, Callable[[Suite, AgentCase, str], Awaitable[_Outcome]]] = {
    "assessment": _assessment,
    "followup": _followup,
    "drug_check": _drug_check,
}


def _langfuse_attributes(case: AgentCase, repeat: int) -> AbstractContextManager[Any]:
    """With Langfuse attached, each run is its own trace named after the case,
    in one session per eval run, so a failing case can be opened there."""
    from medagent.infra import tracing

    if not tracing.langfuse_attached():
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(
        session_id=EVAL_RUN_ID,
        trace_name=case.id,
        tags=[case.category, case.workflow, f"repeat-{repeat}"],
        metadata={"case": case.id, "category": case.category},
    )

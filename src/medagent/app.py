"""FastAPI + WebSocket entrypoint wiring every layer together."""

import math
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.agents.trial_finder_agent import TrialFinderAgent
from medagent.auth.dependencies import get_current_user, get_current_user_ws, require_admin
from medagent.auth.security import create_access_token, hash_password, verify_password
from medagent.auth.store import UserStore
from medagent.core.config import Settings, get_settings
from medagent.core.exceptions import AuthError, MedAgentError, UpstreamError
from medagent.core.models import PatientContext, User
from medagent.health import check_readiness
from medagent.infra.logging import configure_logging, get_logger
from medagent.infra.middleware import ObservabilityMiddleware
from medagent.infra.tracing import configure_tracing, get_tracer, shutdown_tracing
from medagent.llm.registry import ProviderRegistry
from medagent.memory.audit_log import AuditLogStore
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from medagent.tools.custom.interaction_checker import InteractionCheckerTool
from medagent.tools.mcp.healthcare import HealthcareMCPClient
from medagent.tools.mcp.http_base import HttpMCPClient
from medagent.tools.mcp.medical import MedicalMCPClient
from medagent.tools.mcp.research import ResearchMCPClient
from medagent.workflows.checkpointing import build_checkpointer, purge_expired_drafts
from medagent.workflows.drug_check import run_drug_check
from medagent.workflows.followup import (
    ApprovalOutcome,
    build_followup_graph,
    discard_draft,
    draft_followup,
    pending_draft,
    resolve_followup,
    run_followup,
)
from medagent.workflows.patient_assessment import run_patient_assessment

logger = get_logger(__name__)
tracer = get_tracer(__name__)


def _doctor_scope(user: User) -> str | None:
    """Per-doctor data isolation filter: None means unrestricted (admin)."""
    return None if user.role == "admin" else user.id


def _retry_after(exc: MedAgentError) -> float | None:
    """The wait to advertise, only for a 429 (rate limit) that knows it."""
    if exc.status_code != 429:
        return None
    wait: float | None = getattr(exc, "retry_after", None)
    return wait


def _error_payload(exc: MedAgentError) -> dict[str, Any]:
    """Client-facing body for an error. An upstream error's raw message can carry
    account ids or internal hostnames, so it is replaced by fixed text; our own
    errors (bad input, unknown patient) are already written for the client."""
    payload: dict[str, Any] = {
        "detail": exc.public_message() if isinstance(exc, UpstreamError) else str(exc)
    }
    retry_after = _retry_after(exc)
    if retry_after is not None:
        payload["retry_after"] = retry_after
    return payload


def _require_doctor_id(patient: PatientContext) -> str:
    """Every persisted patient has doctor_id stamped at creation (see
    create_patient below) -- this should never actually be None in practice,
    but save_visit must never silently write a NULL doctor_id since the
    per-doctor isolation guarantee depends on every row having one."""
    assert patient.doctor_id is not None, f"Patient {patient.id} is missing doctor_id"
    return patient.doctor_id


class AssessRequest(BaseModel):
    message: str


class DrugCheckRequest(BaseModel):
    patient_id: str
    new_drug: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "doctor"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persistent_store = PersistentStore(settings.postgres_dsn)
        self.patient_store = PatientStore(self.persistent_store)
        self.user_store = UserStore(self.persistent_store)
        self.audit_log = AuditLogStore(self.persistent_store)

        self.llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
        embeddings = EmbeddingModel()
        self.vector_store = VectorStore(settings.chroma_host, settings.chroma_port)
        medical_client = MedicalMCPClient(settings)
        healthcare_client = HealthcareMCPClient(settings)
        research_client = ResearchMCPClient(settings)
        # Held so /ready can probe them and report their circuit state.
        self.mcp_clients: dict[str, HttpMCPClient] = {
            "medical_mcp": medical_client,
            "healthcare_mcp": healthcare_client,
            "research_mcp": research_client,
        }

        self.triage = TriageAgent()
        self.drug_safety = DrugSafetyAgent(self.llm, InteractionCheckerTool(research_client))
        self.evidence = EvidenceAgent(
            self.llm, medical_client, GuidelineRetrieverTool(embeddings, self.vector_store)
        )
        self.trial_finder = TrialFinderAgent(self.llm, healthcare_client)
        self.report = ReportAgent()

        # One long-lived graph with a checkpointer: the WebSocket approval flow
        # pauses in it between the doctor's message and their decision.
        self.checkpointer = build_checkpointer()
        self.followup_graph = build_followup_graph(
            self.triage,
            self.drug_safety,
            self.evidence,
            self.trial_finder,
            self.report,
            self.patient_store,
            checkpointer=self.checkpointer,
        )

    async def init(self) -> None:
        # Schema is owned by Alembic ("alembic upgrade head"), run before
        # startup in any real deployment; this just opens the connection pool.
        await self.persistent_store.init_schema()
        await self._bootstrap_admin()

    async def _bootstrap_admin(self) -> None:
        """Seed exactly one admin account when the users table is empty.

        There is no public registration endpoint -- this is the only way an
        admin account can ever come into existence, so every other account
        must be created by an admin via POST /admin/users.
        """
        username = self.settings.admin_bootstrap_username
        password = self.settings.admin_bootstrap_password
        if not username or not password:
            return
        if await self.user_store.count_users() > 0:
            return

        await self.user_store.create_user(username, hash_password(password), role="admin")
        logger.info("admin_bootstrapped", username=username)

    async def close(self) -> None:
        await self.persistent_store.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(resolved_settings.log_level)
        configure_tracing(resolved_settings)
        if resolved_settings.langchain_tracing_v2:
            logger.warning(
                "langsmith_tracing_enabled",
                note="prompts and graph state, which contain patient data, are sent to "
                "LangSmith; use only with synthetic data",
            )
        state = AppState(resolved_settings)
        await state.init()
        app.state.medagent = state
        logger.info("medagent_startup_complete")
        yield
        await state.close()
        shutdown_tracing()
        logger.info("medagent_shutdown_complete")

    app = FastAPI(title="MedAgent", lifespan=lifespan)
    app.add_middleware(ObservabilityMiddleware)

    @app.exception_handler(MedAgentError)
    async def _medagent_error_handler(_request: Request, exc: MedAgentError) -> JSONResponse:
        headers = {}
        retry_after = _retry_after(exc)
        if retry_after is not None:
            headers["Retry-After"] = str(math.ceil(retry_after))
        return JSONResponse(
            status_code=exc.status_code, content=_error_payload(exc), headers=headers
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        state: AppState = app.state.medagent
        circuits = {name: client.breaker_state for name, client in state.mcp_clients.items()}
        llm_circuit = getattr(state.llm, "circuit_state", None)
        if llm_circuit is not None:
            circuits["llm"] = llm_circuit
        result = await check_readiness(
            postgres=state.persistent_store.ping,
            chroma=state.vector_store.ping,
            mcp={name: client.probe for name, client in state.mcp_clients.items()},
            circuits=circuits,
        )
        return JSONResponse(
            status_code=503 if result.status == "down" else 200,
            content={"status": result.status, "checks": result.checks},
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/users", response_model=User)
    async def create_user(body: CreateUserRequest, _admin: User = Depends(require_admin)) -> User:
        # No public registration endpoint: every account (after the one admin
        # seeded from ADMIN_BOOTSTRAP_USERNAME/PASSWORD at startup) is created
        # by an existing admin, so patient data is never open to self-signup.
        state: AppState = app.state.medagent
        try:
            return await state.user_store.create_user(
                body.username, hash_password(body.password), body.role
            )
        except AuthError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/auth/token", response_model=Token)
    async def login(form: OAuth2PasswordRequestForm = Depends()) -> Token:
        state: AppState = app.state.medagent
        record = await state.user_store.get_by_username(form.username)
        if record is None or not verify_password(form.password, record[1]):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        token = create_access_token(
            form.username,
            state.settings.jwt_secret_key,
            state.settings.jwt_algorithm,
            state.settings.jwt_expire_minutes,
        )
        return Token(access_token=token)

    @app.post("/patients", response_model=PatientContext)
    async def create_patient(
        context: PatientContext, user: User = Depends(get_current_user)
    ) -> PatientContext:
        state: AppState = app.state.medagent
        # Server-stamped, never client-supplied: whatever doctor_id the
        # request body carried is overwritten with the creating doctor's own
        # identity, and it's immutable after this (see PatientStore.save_patient).
        context.doctor_id = user.id
        await state.patient_store.save_patient(context)
        await state.audit_log.record(user, context.id, "patient_created")
        return context

    @app.get("/patients/{patient_id}", response_model=PatientContext)
    async def get_patient(
        patient_id: str, user: User = Depends(get_current_user)
    ) -> PatientContext:
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id, _doctor_scope(user))
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        await state.audit_log.record(user, patient_id, "patient_viewed")
        return patient

    @app.get("/patients/{patient_id}/audit-log")
    async def get_patient_audit_log(
        patient_id: str, user: User = Depends(get_current_user)
    ) -> list[dict[str, Any]]:
        state: AppState = app.state.medagent
        # Audit log itself isn't doctor-filtered, so ownership must be
        # checked explicitly here before returning any rows.
        patient = await state.patient_store.get_patient(patient_id, _doctor_scope(user))
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        await state.audit_log.record(user, patient_id, "audit_log_viewed")
        return await state.audit_log.for_patient(patient_id)

    @app.post("/patients/{patient_id}/assess")
    async def assess_patient(
        patient_id: str, body: AssessRequest, user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        # REST has no interactive approval channel, so it auto-saves. The
        # human-in-the-loop checkpoint lives in the WebSocket chat below.
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id, _doctor_scope(user))
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        report = await run_patient_assessment(
            state.triage,
            state.drug_safety,
            state.evidence,
            state.trial_finder,
            state.report,
            patient,
            body.message,
        )
        await state.patient_store.save_patient(patient)
        await state.patient_store.save_visit(
            patient_id, _require_doctor_id(patient), body.message, report
        )
        await state.audit_log.record(user, patient_id, "assessment_run", {"message": body.message})
        return {"report": report}

    @app.post("/patients/{patient_id}/followup")
    async def followup_visit(
        patient_id: str, body: AssessRequest, user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        state: AppState = app.state.medagent
        report, patient = await run_followup(
            state.triage,
            state.drug_safety,
            state.evidence,
            state.trial_finder,
            state.report,
            state.patient_store,
            patient_id,
            body.message,
            doctor_id=_doctor_scope(user),
        )
        await state.patient_store.save_patient(patient)
        await state.patient_store.save_visit(
            patient_id, _require_doctor_id(patient), body.message, report
        )
        await state.audit_log.record(user, patient_id, "followup_run", {"message": body.message})
        return {"report": report}

    @app.post("/drug-check")
    async def drug_check(
        body: DrugCheckRequest, user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        state: AppState = app.state.medagent
        answer = await run_drug_check(
            state.drug_safety,
            state.patient_store,
            body.patient_id,
            body.new_drug,
            doctor_id=_doctor_scope(user),
        )
        await state.audit_log.record(
            user, body.patient_id, "drug_check_run", {"new_drug": body.new_drug}
        )
        return {"answer": answer}

    @app.websocket("/ws/{patient_id}")
    async def chat(websocket: WebSocket, patient_id: str) -> None:
        """Interactive chat with a human-in-the-loop approval gate.

        Each doctor message drafts a report but does NOT save it. While it
        runs the server streams {"type": "progress", "step": ..., "status": ...}
        frames, then sends {"type": "pending_approval", "report": ...} and waits
        for a decision: "approve" saves it verbatim, "reject" discards it, and
        any other text is treated as the doctor's edited version of the report
        and saved in its place.

        A draft survives a dropped connection. Reconnect with ?resume=1 to get
        it back ("resumed": true) and decide; connecting without it discards
        any stale draft, so a fresh question is never mistaken for an edit.
        """
        state: AppState = app.state.medagent
        user = await get_current_user_ws(websocket)
        if user is None:
            await websocket.close(code=1008, reason="Not authenticated")
            return

        graph = state.followup_graph
        thread_id = f"{user.id}:{patient_id}"
        await websocket.accept()
        # The HTTP middleware doesn't see WebSockets: correlate this session's
        # log lines with an id of its own.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=uuid.uuid4().hex)
        try:
            await purge_expired_drafts(
                state.checkpointer, timedelta(minutes=state.settings.approval_draft_ttl_minutes)
            )
            stale = await pending_draft(graph, thread_id)
            if stale is not None:
                if websocket.query_params.get("resume") in ("1", "true"):
                    await state.audit_log.record(user, patient_id, "report_draft_resumed")
                    await websocket.send_json(
                        {"type": "pending_approval", "report": stale.report, "resumed": True}
                    )
                    await _await_decision(websocket, state, user, patient_id, thread_id)
                else:
                    await discard_draft(graph, thread_id)
                    await state.audit_log.record(user, patient_id, "report_draft_discarded")

            while True:
                message = await websocket.receive_text()
                try:
                    with tracer.start_as_current_span("ws.draft"):
                        draft = await draft_followup(
                            graph,
                            patient_id=patient_id,
                            message=message,
                            doctor_id=_doctor_scope(user),
                            thread_id=thread_id,
                            on_progress=websocket.send_json,
                        )
                except MedAgentError as exc:
                    await websocket.send_json({"type": "error", **_error_payload(exc)})
                    continue

                await state.audit_log.record(
                    user, patient_id, "report_drafted", {"message": message}
                )
                await websocket.send_json({"type": "pending_approval", "report": draft.report})
                await _await_decision(websocket, state, user, patient_id, thread_id)
        except WebSocketDisconnect:
            # A pending draft is deliberately left in place for ?resume=1.
            logger.info("websocket_disconnected")

    async def _await_decision(
        websocket: WebSocket, state: AppState, user: User, patient_id: str, thread_id: str
    ) -> None:
        decision = (await websocket.receive_text()).strip()
        outcome = await resolve_followup(
            state.followup_graph, thread_id=thread_id, decision=decision
        )
        await _apply_outcome(websocket, state, user, patient_id, outcome)

    async def _apply_outcome(
        websocket: WebSocket,
        state: AppState,
        user: User,
        patient_id: str,
        outcome: ApprovalOutcome,
    ) -> None:
        if outcome.kind == "rejected":
            await state.audit_log.record(user, patient_id, "report_rejected")
            await websocket.send_json({"type": "rejected"})
            return

        assert outcome.report is not None
        await state.patient_store.save_patient(outcome.context)
        await state.patient_store.save_visit(
            patient_id, _require_doctor_id(outcome.context), outcome.message, outcome.report
        )
        if outcome.kind == "approved":
            await state.audit_log.record(user, patient_id, "report_approved")
        else:
            await state.audit_log.record(
                user, patient_id, "report_edited", {"edited_report": outcome.report}
            )
        await websocket.send_json({"type": "saved", "report": outcome.report})

    return app


app = create_app()

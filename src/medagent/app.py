"""FastAPI + WebSocket entrypoint wiring every layer together."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

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
from medagent.core.exceptions import AuthError, MedAgentError
from medagent.core.models import PatientContext, User
from medagent.infra.logging import configure_logging, get_logger
from medagent.infra.middleware import ObservabilityMiddleware
from medagent.infra.tracing import configure_tracing
from medagent.llm.registry import ProviderRegistry
from medagent.memory.audit_log import AuditLogStore
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from medagent.tools.custom.interaction_checker import InteractionCheckerTool
from medagent.tools.mcp.healthcare import HealthcareMCPClient
from medagent.tools.mcp.medical import MedicalMCPClient
from medagent.workflows.drug_check import run_drug_check
from medagent.workflows.followup import run_followup
from medagent.workflows.patient_assessment import run_patient_assessment

logger = get_logger(__name__)


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

        llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
        embeddings = EmbeddingModel()
        vector_store = VectorStore(settings.chroma_host, settings.chroma_port)

        self.triage = TriageAgent()
        self.drug_safety = DrugSafetyAgent(llm, InteractionCheckerTool())
        self.evidence = EvidenceAgent(
            llm, MedicalMCPClient(settings), GuidelineRetrieverTool(embeddings, vector_store)
        )
        self.trial_finder = TrialFinderAgent(llm, HealthcareMCPClient(settings))
        self.report = ReportAgent()

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
        configure_tracing()
        state = AppState(resolved_settings)
        await state.init()
        app.state.medagent = state
        logger.info("medagent_startup_complete")
        yield
        await state.close()
        logger.info("medagent_shutdown_complete")

    app = FastAPI(title="MedAgent", lifespan=lifespan)
    app.add_middleware(ObservabilityMiddleware)

    @app.exception_handler(MedAgentError)
    async def _medagent_error_handler(_request: Request, exc: MedAgentError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
        await state.patient_store.save_patient(context)
        await state.audit_log.record(user, context.id, "patient_created")
        return context

    @app.get("/patients/{patient_id}", response_model=PatientContext)
    async def get_patient(
        patient_id: str, user: User = Depends(get_current_user)
    ) -> PatientContext:
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        await state.audit_log.record(user, patient_id, "patient_viewed")
        return patient

    @app.get("/patients/{patient_id}/audit-log")
    async def get_patient_audit_log(
        patient_id: str, user: User = Depends(get_current_user)
    ) -> list[dict[str, Any]]:
        state: AppState = app.state.medagent
        await state.audit_log.record(user, patient_id, "audit_log_viewed")
        return await state.audit_log.for_patient(patient_id)

    @app.post("/patients/{patient_id}/assess")
    async def assess_patient(
        patient_id: str, body: AssessRequest, user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        # REST has no interactive approval channel, so it auto-saves. The
        # human-in-the-loop checkpoint lives in the WebSocket chat below.
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id)
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
        )
        await state.patient_store.save_patient(patient)
        await state.audit_log.record(user, patient_id, "followup_run", {"message": body.message})
        return {"report": report}

    @app.post("/drug-check")
    async def drug_check(
        body: DrugCheckRequest, user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        state: AppState = app.state.medagent
        answer = await run_drug_check(
            state.drug_safety, state.patient_store, body.patient_id, body.new_drug
        )
        await state.audit_log.record(
            user, body.patient_id, "drug_check_run", {"new_drug": body.new_drug}
        )
        return {"answer": answer}

    @app.websocket("/ws/{patient_id}")
    async def chat(websocket: WebSocket, patient_id: str) -> None:
        """Interactive chat with a human-in-the-loop approval gate.

        Each doctor message drafts a report but does NOT save it. The server
        sends {"type": "pending_approval", "report": ...} and waits for a
        decision: "approve" saves it verbatim, "reject" discards it, and any
        other text is treated as the doctor's edited version of the report
        and saved in its place.
        """
        state: AppState = app.state.medagent
        user = await get_current_user_ws(websocket)
        if user is None:
            await websocket.close(code=1008, reason="Not authenticated")
            return

        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    report, patient = await run_followup(
                        state.triage,
                        state.drug_safety,
                        state.evidence,
                        state.trial_finder,
                        state.report,
                        state.patient_store,
                        patient_id,
                        message,
                    )
                except MedAgentError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

                await state.audit_log.record(
                    user, patient_id, "report_drafted", {"message": message}
                )
                await websocket.send_json({"type": "pending_approval", "report": report})
                decision = (await websocket.receive_text()).strip()

                if decision.lower() == "approve":
                    await state.patient_store.save_patient(patient)
                    await state.audit_log.record(user, patient_id, "report_approved")
                    await websocket.send_json({"type": "saved", "report": report})
                elif decision.lower() == "reject":
                    await state.audit_log.record(user, patient_id, "report_rejected")
                    await websocket.send_json({"type": "rejected"})
                else:
                    await state.patient_store.save_patient(patient)
                    await state.audit_log.record(
                        user, patient_id, "report_edited", {"edited_report": decision}
                    )
                    await websocket.send_json({"type": "saved", "report": decision})
        except WebSocketDisconnect:
            logger.info("websocket_disconnected", patient_id=patient_id)

    return app


app = create_app()

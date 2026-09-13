"""FastAPI + WebSocket entrypoint wiring every layer together."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.auth.dependencies import get_current_user, get_current_user_ws
from medagent.auth.security import create_access_token, hash_password, verify_password
from medagent.auth.store import UserStore
from medagent.core.config import Settings, get_settings
from medagent.core.exceptions import AuthError, MedAgentError
from medagent.core.models import PatientContext, User
from medagent.infra.logging import configure_logging, get_logger
from medagent.infra.middleware import ObservabilityMiddleware
from medagent.infra.tracing import configure_tracing
from medagent.llm.registry import ProviderRegistry
from medagent.memory.patient_store import PatientStore
from medagent.memory.persistent import PersistentStore
from medagent.rag.embeddings import EmbeddingModel
from medagent.rag.retriever import GuidelineRetrieverTool
from medagent.rag.vector_store import VectorStore
from medagent.tools.custom.interaction_checker import InteractionCheckerTool
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


class RegisterRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persistent_store = PersistentStore(settings.postgres_dsn)
        self.patient_store = PatientStore(self.persistent_store)
        self.user_store = UserStore(self.persistent_store)

        llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
        embeddings = EmbeddingModel()
        vector_store = VectorStore(settings.chroma_persist_dir)

        self.triage = TriageAgent()
        self.drug_safety = DrugSafetyAgent(llm, InteractionCheckerTool())
        self.evidence = EvidenceAgent(
            llm, MedicalMCPClient(settings), GuidelineRetrieverTool(embeddings, vector_store)
        )
        self.report = ReportAgent()

    async def init(self) -> None:
        # Schema is owned by Alembic ("alembic upgrade head"), run before
        # startup in any real deployment; this just opens the connection pool.
        await self.persistent_store.init_schema()

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
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/auth/register", response_model=User)
    async def register(body: RegisterRequest) -> User:
        state: AppState = app.state.medagent
        try:
            return await state.user_store.create_user(body.username, hash_password(body.password))
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
        context: PatientContext, _user: User = Depends(get_current_user)
    ) -> PatientContext:
        state: AppState = app.state.medagent
        await state.patient_store.save_patient(context)
        return context

    @app.get("/patients/{patient_id}", response_model=PatientContext)
    async def get_patient(
        patient_id: str, _user: User = Depends(get_current_user)
    ) -> PatientContext:
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        return patient

    @app.post("/patients/{patient_id}/assess")
    async def assess_patient(
        patient_id: str, body: AssessRequest, _user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        # REST has no interactive approval channel, so it auto-saves. The
        # human-in-the-loop checkpoint lives in the WebSocket chat below.
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        report = await run_patient_assessment(
            state.triage, state.drug_safety, state.evidence, state.report, patient, body.message
        )
        await state.patient_store.save_patient(patient)
        return {"report": report}

    @app.post("/patients/{patient_id}/followup")
    async def followup_visit(
        patient_id: str, body: AssessRequest, _user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        state: AppState = app.state.medagent
        report, patient = await run_followup(
            state.triage,
            state.drug_safety,
            state.evidence,
            state.report,
            state.patient_store,
            patient_id,
            body.message,
        )
        await state.patient_store.save_patient(patient)
        return {"report": report}

    @app.post("/drug-check")
    async def drug_check(
        body: DrugCheckRequest, _user: User = Depends(get_current_user)
    ) -> dict[str, str]:
        state: AppState = app.state.medagent
        answer = await run_drug_check(
            state.drug_safety, state.patient_store, body.patient_id, body.new_drug
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
                        state.report,
                        state.patient_store,
                        patient_id,
                        message,
                    )
                except MedAgentError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

                await websocket.send_json({"type": "pending_approval", "report": report})
                decision = (await websocket.receive_text()).strip()

                if decision.lower() == "approve":
                    await state.patient_store.save_patient(patient)
                    await websocket.send_json({"type": "saved", "report": report})
                elif decision.lower() == "reject":
                    await websocket.send_json({"type": "rejected"})
                else:
                    await state.patient_store.save_patient(patient)
                    await websocket.send_json({"type": "saved", "report": decision})
        except WebSocketDisconnect:
            logger.info("websocket_disconnected", patient_id=patient_id)

    return app


app = create_app()

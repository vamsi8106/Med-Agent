"""FastAPI + WebSocket entrypoint wiring every layer together."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.agents.evidence_agent import EvidenceAgent
from medagent.agents.report_agent import ReportAgent
from medagent.agents.triage_agent import TriageAgent
from medagent.core.config import Settings, get_settings
from medagent.core.exceptions import MedAgentError
from medagent.core.models import PatientContext
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


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.persistent_store = PersistentStore(settings.database_path)
        self.patient_store = PatientStore(self.persistent_store)

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
        await self.persistent_store.init_schema()


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

    @app.post("/patients", response_model=PatientContext)
    async def create_patient(context: PatientContext) -> PatientContext:
        state: AppState = app.state.medagent
        await state.patient_store.save_patient(context)
        return context

    @app.get("/patients/{patient_id}", response_model=PatientContext)
    async def get_patient(patient_id: str) -> PatientContext:
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        return patient

    @app.post("/patients/{patient_id}/assess")
    async def assess_patient(patient_id: str, body: AssessRequest) -> dict[str, str]:
        state: AppState = app.state.medagent
        patient = await state.patient_store.get_patient(patient_id)
        if patient is None:
            raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
        report = await run_patient_assessment(
            state.triage,
            state.drug_safety,
            state.evidence,
            state.report,
            state.patient_store,
            patient,
            body.message,
        )
        return {"report": report}

    @app.post("/patients/{patient_id}/followup")
    async def followup_visit(patient_id: str, body: AssessRequest) -> dict[str, str]:
        state: AppState = app.state.medagent
        report = await run_followup(
            state.triage,
            state.drug_safety,
            state.evidence,
            state.report,
            state.patient_store,
            patient_id,
            body.message,
        )
        return {"report": report}

    @app.post("/drug-check")
    async def drug_check(body: DrugCheckRequest) -> dict[str, str]:
        state: AppState = app.state.medagent
        answer = await run_drug_check(
            state.drug_safety, state.patient_store, body.patient_id, body.new_drug
        )
        return {"answer": answer}

    @app.websocket("/ws/{patient_id}")
    async def chat(websocket: WebSocket, patient_id: str) -> None:
        state: AppState = app.state.medagent
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    report = await run_followup(
                        state.triage,
                        state.drug_safety,
                        state.evidence,
                        state.report,
                        state.patient_store,
                        patient_id,
                        message,
                    )
                    await websocket.send_text(report)
                except MedAgentError as exc:
                    await websocket.send_text(f"error: {exc}")
        except WebSocketDisconnect:
            logger.info("websocket_disconnected", patient_id=patient_id)

    return app


app = create_app()

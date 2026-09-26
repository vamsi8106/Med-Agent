"""FastAPI entrypoint: lifespan, middleware and routers. Routes live in `api/`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from medagent.api.errors import medagent_error_handler
from medagent.api.routes import auth, patients, system, ws
from medagent.api.state import AppState
from medagent.core.config import Settings, get_settings
from medagent.core.exceptions import MedAgentError
from medagent.infra.logging import configure_logging, get_logger
from medagent.infra.middleware import ObservabilityMiddleware
from medagent.infra.tracing import configure_tracing, shutdown_tracing

logger = get_logger(__name__)


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
    app.add_exception_handler(MedAgentError, medagent_error_handler)  # type: ignore[arg-type]
    for router in (system.router, auth.router, patients.router, ws.router):
        app.include_router(router)
    return app


app = create_app()

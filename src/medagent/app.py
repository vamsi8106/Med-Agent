"""FastAPI entrypoint: lifespan, middleware and routers. Routes live in `api/`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from medagent.api.errors import medagent_error_handler
from medagent.api.routes import auth, patients, system, ws
from medagent.api.state import AppState
from medagent.core.config import INSECURE_JWT_SECRET, Settings, get_settings
from medagent.core.exceptions import MedAgentError
from medagent.infra.logging import configure_logging, get_logger
from medagent.infra.middleware import ObservabilityMiddleware, SecurityMiddleware
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
        if resolved_settings.jwt_secret_key == INSECURE_JWT_SECRET:
            logger.warning(
                "insecure_jwt_secret",
                note="JWT_SECRET_KEY is the public placeholder; anyone can forge tokens. "
                "APP_ENV=production refuses to start like this.",
            )
        state = AppState(resolved_settings)
        await state.init()
        app.state.medagent = state
        logger.info("medagent_startup_complete")
        yield
        await state.close()
        shutdown_tracing()
        logger.info("medagent_shutdown_complete")

    production = resolved_settings.app_env == "production"
    # The interactive docs map every route and schema; not served in production.
    app = FastAPI(
        title="MedAgent",
        lifespan=lifespan,
        docs_url=None if production else "/docs",
        redoc_url=None if production else "/redoc",
        openapi_url=None if production else "/openapi.json",
    )
    # Added first = innermost: the observability middleware still logs and
    # counts a 413 that the security middleware returns.
    app.add_middleware(
        SecurityMiddleware,
        max_body_bytes=resolved_settings.max_request_body_bytes,
        hsts=production,
    )
    app.add_middleware(ObservabilityMiddleware)
    app.add_exception_handler(MedAgentError, medagent_error_handler)  # type: ignore[arg-type]
    for router in (system.router, auth.router, patients.router, ws.router):
        app.include_router(router)
    return app


app = create_app()

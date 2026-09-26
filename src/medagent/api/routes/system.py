"""Liveness, readiness and metrics."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from medagent.api.state import AppState, get_state
from medagent.health import check_readiness

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(state: AppState = Depends(get_state)) -> JSONResponse:
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


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

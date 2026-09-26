"""Integration test against the real medical-mcp server, via its HTTP bridge
sidecar (Phase 2 gate, updated for Phase A's containerized MCP topology).

MedicalMCPClient no longer spawns `npx` itself -- it talks to the
medical-mcp-bridge container over HTTP (see docker/medical-mcp-bridge), which
spawns the real medical-mcp process internally over stdio. Not run as part of
`make unit-tests`; run explicitly with `uv run pytest tests/integration`
against a running `docker compose up medical-mcp-bridge`.
"""

import httpx
import pytest

from medagent.core.config import get_settings
from medagent.tools.mcp.medical import MedicalMCPClient


def _bridge_reachable() -> bool:
    try:
        response = httpx.get(f"{get_settings().medical_mcp_url}/health", timeout=2.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = pytest.mark.skipif(
    not _bridge_reachable(), reason="medical-mcp-bridge is not reachable in this environment"
)


async def test_search_drugs_metformin_returns_structured_response() -> None:
    async with MedicalMCPClient() as client:
        content = await client.search_drugs("metformin")

    assert content is not None

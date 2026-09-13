"""Integration test against the real medical-mcp server (Phase 2 gate).

Requires network access and `npx` on PATH to fetch/run `medical-mcp`. Not run as
part of `make unit-tests`; run explicitly with `uv run pytest tests/integration`.
"""

import shutil

import pytest

from medagent.tools.mcp.medical import MedicalMCPClient

pytestmark = pytest.mark.skipif(
    shutil.which("npx") is None, reason="npx not available in this environment"
)


async def test_search_drugs_metformin_returns_structured_response() -> None:
    async with MedicalMCPClient() as client:
        content = await client.search_drugs("metformin")

    assert content is not None

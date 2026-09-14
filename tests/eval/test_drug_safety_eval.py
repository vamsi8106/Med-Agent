"""Live golden-dataset eval for DrugSafetyAgent: real Groq LLM + real MCP servers.

Not part of the unit suite: needs a live GROQ_API_KEY and working MCP servers.
Run deliberately via `make eval`, never as part of `make pre-commit`/CI.
"""

import pytest

from medagent.agents.drug_safety_agent import DrugSafetyAgent
from medagent.core.config import get_settings
from medagent.core.exceptions import MCPError
from medagent.core.types import InteractionSeverity
from medagent.llm.registry import ProviderRegistry
from medagent.tools.custom.interaction_checker import InteractionCheckerTool
from medagent.tools.mcp.research import ResearchMCPClient
from tests.eval.golden_dataset import CASES, GoldenCase

pytestmark = pytest.mark.skipif(
    not get_settings().groq_api_key, reason="GROQ_API_KEY not set; skipping live eval"
)

_SEVERITY_RANK: dict[InteractionSeverity, int] = {
    InteractionSeverity.NONE: 0,
    InteractionSeverity.MINOR: 1,
    InteractionSeverity.MODERATE: 2,
    InteractionSeverity.MAJOR: 3,
    InteractionSeverity.CONTRAINDICATED: 4,
}


@pytest.fixture
async def drug_safety_agent() -> DrugSafetyAgent:
    settings = get_settings()
    llm = ProviderRegistry.get_provider(settings.llm_provider, settings)
    research_client = ResearchMCPClient(settings)
    try:
        async with research_client as client:
            # __aenter__ only opens an httpx.AsyncClient (no network call by
            # itself, unlike the old stdio transport's subprocess handshake),
            # so a real request is needed here to actually detect a down
            # server rather than silently skipping the connectivity check.
            await client.drug_safety_profile("aspirin")
    except MCPError as exc:
        pytest.skip(f"MCP server unreachable, skipping live eval: {exc}")

    interaction_checker = InteractionCheckerTool(research_client)
    return DrugSafetyAgent(llm=llm, interaction_checker=interaction_checker)


@pytest.mark.parametrize("case", CASES, ids=[",".join(c.drugs) for c in CASES])
async def test_known_interaction_meets_minimum_severity(
    drug_safety_agent: DrugSafetyAgent, case: GoldenCase
) -> None:
    message = f"Check interactions for {', '.join(case.drugs)}"
    result = await drug_safety_agent.run_result(case.patient, message)

    matching = [i for i in result.interactions if {i.drug_a, i.drug_b} == set(case.drugs)]
    assert matching, f"No interaction reported for {case.drugs} ({case.notes})"

    actual_rank = max(_SEVERITY_RANK[i.severity] for i in matching)
    expected_rank = _SEVERITY_RANK[case.min_expected_severity]
    assert actual_rank >= expected_rank, (
        f"{case.drugs} reported as {[i.severity for i in matching]}, "
        f"expected at least {case.min_expected_severity} ({case.notes})"
    )

    assert result.summary.strip()
    for drug in case.drugs:
        assert drug.lower() in result.summary.lower()

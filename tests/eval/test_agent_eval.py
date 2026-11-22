"""Agent-eval gates, part of `make eval` (and CI on PRs into main): every case
runs end to end once, then safety must be perfect and quality must stay
within its floors and ceilings (agent_thresholds.py). Once, not three times:
a full repeated run costs about a day of Groq free-tier quota. Repeats, the
report and the diff are `make agent-eval`, run by hand before big changes.

Skips without an LLM key, the MCP servers or Docker.
"""

import pytest
import pytest_asyncio

from medagent.core.config import get_settings
from medagent.core.exceptions import MCPError
from medagent.tools.mcp.medical import MedicalMCPClient
from tests.eval.agent_cases import ALL_CASES
from tests.eval.agent_eval import evaluate, format_report
from tests.eval.agent_metrics import Summary
from tests.eval.agent_thresholds import CEILINGS, FLOORS, SAFETY_GATES

pytestmark = pytest.mark.skipif(
    get_settings().llm_provider == "groq" and not get_settings().groq_api_key,
    reason="GROQ_API_KEY not set; skipping live eval",
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def summary(postgres_dsn: str) -> Summary:
    try:
        async with MedicalMCPClient(get_settings()) as client:
            await client.search_drugs("aspirin")
    except MCPError as exc:
        pytest.skip(f"medical-mcp-bridge unreachable, skipping live eval: {exc}")
    _, result, model = await evaluate(postgres_dsn, ALL_CASES, lambda _case: 1)
    print("\n" + format_report(result, model, 0.0, None))
    return result


def test_the_run_is_conclusive(summary: Summary) -> None:
    assert not summary.inconclusive, (
        f"{'; '.join(summary.inconclusive_reasons)} {summary.counts.get('infra_reasons')}: "
        "re-run when the LLM/MCP servers are healthy"
    )


@pytest.mark.parametrize("gate", SAFETY_GATES)
def test_safety_gate_holds_in_every_run(summary: Summary, gate: str) -> None:
    rate = summary.rates[gate]
    assert rate.total, f"no run exercised {gate}"
    assert rate.hits == rate.total, f"{gate}: {rate.hits}/{rate.total} -- {summary.failures}"


@pytest.mark.parametrize("metric", sorted(FLOORS))
def test_quality_stays_above_its_floor(summary: Summary, metric: str) -> None:
    value = summary.rates[metric].value
    if value is None:
        pytest.skip(f"nothing exercised {metric}")
    assert value >= FLOORS[metric], f"{metric} {value:.0%} < floor {FLOORS[metric]:.0%}"


@pytest.mark.parametrize("metric", sorted(CEILINGS))
def test_errors_stay_below_their_ceiling(summary: Summary, metric: str) -> None:
    value = summary.rates[metric].value
    if value is None:
        pytest.skip(f"nothing exercised {metric}")
    assert value <= CEILINGS[metric], f"{metric} {value:.0%} > ceiling {CEILINGS[metric]:.0%}"

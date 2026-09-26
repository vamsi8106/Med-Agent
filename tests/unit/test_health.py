import asyncio

import pytest

from medagent import health
from medagent.health import check_readiness


async def _ok() -> bool:
    return True


async def _down() -> bool:
    return False


async def _explodes() -> bool:
    raise RuntimeError("could not connect to internal-db.corp:5432 as admin password=hunter2")


async def _hangs() -> bool:
    await asyncio.sleep(30)
    return True


async def _ready(**overrides):
    kwargs = {
        "postgres": _ok,
        "chroma": _ok,
        "mcp": {"medical_mcp": _ok, "research_mcp": _ok},
        "circuits": {"llm": "closed", "medical_mcp": "closed"},
    }
    kwargs.update(overrides)
    return await check_readiness(**kwargs)


async def test_everything_healthy_is_ok() -> None:
    result = await _ready()
    assert result.status == "ok"
    assert result.checks["postgres"] == "ok"
    assert result.checks["llm_circuit"] == "closed"


async def test_postgres_down_means_not_ready() -> None:
    result = await _ready(postgres=_down)
    assert result.status == "down"
    assert result.checks["postgres"] == "down"


@pytest.mark.parametrize(
    "overrides",
    [
        {"chroma": _down},
        {"mcp": {"medical_mcp": _down, "research_mcp": _ok}},
        {"circuits": {"llm": "open", "medical_mcp": "closed"}},
        {"circuits": {"llm": "half_open", "medical_mcp": "closed"}},
    ],
)
async def test_a_non_essential_dependency_problem_degrades_but_does_not_fail(
    overrides: dict,
) -> None:
    """Guideline search, an MCP server, or the LLM being out yields partial
    reports, not no service -- so the instance stays in rotation."""
    assert (await _ready(**overrides)).status == "degraded"


async def test_a_probe_that_raises_is_down_and_its_error_text_is_never_reported() -> None:
    result = await _ready(chroma=_explodes)

    assert result.checks["chroma"] == "down"
    assert "hunter2" not in repr(result)
    assert "internal-db" not in repr(result)


async def test_a_hung_dependency_times_out_instead_of_hanging_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health, "PROBE_TIMEOUT_SECONDS", 0.05)

    started = asyncio.get_running_loop().time()
    result = await _ready(chroma=_hangs)

    assert result.checks["chroma"] == "down"
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_probes_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_ok() -> bool:
        await asyncio.sleep(0.2)
        return True

    started = asyncio.get_running_loop().time()
    result = await _ready(
        postgres=slow_ok, chroma=slow_ok, mcp={"a": slow_ok, "b": slow_ok, "c": slow_ok}
    )

    assert result.status == "ok"
    assert asyncio.get_running_loop().time() - started < 0.5  # five sequential would be 1.0s

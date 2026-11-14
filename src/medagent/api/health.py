"""Readiness for /ready: which dependencies this instance can actually use.

Liveness (/health) only says the process is up -- and is what the Docker
healthcheck uses, because restarting the app cannot fix a down dependency.
Readiness says whether it can do useful work.

Reports only ok / degraded / down per dependency. The endpoint is
unauthenticated, so it never includes URLs, hostnames or error text. The LLM is
reported by its circuit-breaker state rather than probed: a probe would spend
tokens and add rate-limit pressure on every load-balancer poll.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

PROBE_TIMEOUT_SECONDS = 2.0

Probe = Callable[[], Awaitable[bool]]


@dataclass
class Readiness:
    status: str  # "ok" | "degraded" | "down"
    checks: dict[str, str]


async def _probe(probe: Probe) -> str:
    try:
        healthy = await asyncio.wait_for(probe(), timeout=PROBE_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 - any failure, including a timeout, means "down"
        return "down"
    return "ok" if healthy else "down"


async def check_readiness(
    *, postgres: Probe, chroma: Probe, mcp: dict[str, Probe], circuits: dict[str, str]
) -> Readiness:
    """Postgres is required (down -> the instance is not ready, 503). Everything
    else degrades: guideline search, an MCP server, or the LLM being unavailable
    yields partial reports rather than no service."""
    probes: dict[str, Probe] = {"postgres": postgres, "chroma": chroma, **mcp}
    results = await asyncio.gather(*(_probe(p) for p in probes.values()))
    checks = dict(zip(probes, results, strict=True))
    for name, state in circuits.items():
        checks[f"{name}_circuit"] = state

    if checks["postgres"] == "down":
        status = "down"
    elif any(v == "down" for v in checks.values()) or any(
        state != "closed" for state in circuits.values()
    ):
        status = "degraded"
    else:
        status = "ok"
    return Readiness(status=status, checks=checks)

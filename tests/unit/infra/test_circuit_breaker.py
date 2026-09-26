import pytest

from medagent.core.exceptions import MCPError
from medagent.infra.circuit_breaker import CircuitBreaker, CircuitState


async def test_circuit_stays_closed_on_success() -> None:
    breaker = CircuitBreaker(failure_threshold=2)

    async def ok() -> str:
        return "ok"

    assert await breaker.call(ok) == "ok"
    assert breaker.state is CircuitState.CLOSED


async def test_circuit_opens_after_threshold_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=60)

    async def fail() -> None:
        raise ValueError("boom")

    for _ in range(2):
        with pytest.raises(ValueError):
            await breaker.call(fail)

    assert breaker.state is CircuitState.OPEN
    with pytest.raises(MCPError):
        await breaker.call(fail)


async def test_circuit_half_opens_after_recovery_timeout() -> None:
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)

    async def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await breaker.call(fail)
    assert breaker.state is CircuitState.OPEN

    import asyncio

    await asyncio.sleep(0.02)
    assert breaker.state is CircuitState.HALF_OPEN


async def test_success_resets_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=2)

    async def fail() -> None:
        raise ValueError("boom")

    async def ok() -> str:
        return "ok"

    with pytest.raises(ValueError):
        await breaker.call(fail)
    await breaker.call(ok)
    assert breaker.state is CircuitState.CLOSED

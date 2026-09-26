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


# --- what counts as a failure, and who is told ---------------------------------------

from medagent.core.exceptions import ProviderError  # noqa: E402
from tests.conftest import metric_value  # noqa: E402


async def _fail_with(error: Exception) -> None:
    raise error


async def test_client_errors_and_rate_limits_never_open_the_breaker() -> None:
    """A burst of bad requests (4xx) or a rate limit means the server is up and
    answering; counting them would take a healthy dependency offline."""
    breaker = CircuitBreaker(failure_threshold=2, name="test-neutral", error_type=ProviderError)

    for error in (
        ProviderError("bad", reason="client_error"),
        ProviderError("limited", reason="rate_limited", retryable=True),
    ) * 5:
        with pytest.raises(ProviderError):
            await breaker.call(_fail_with, error)

    assert breaker.state is CircuitState.CLOSED


@pytest.mark.parametrize("reason", ["timeout", "server_error", "connection_error"])
async def test_genuine_upstream_trouble_opens_the_breaker(reason: str) -> None:
    breaker = CircuitBreaker(failure_threshold=2, name="test-trips", error_type=ProviderError)

    for _ in range(2):
        with pytest.raises(ProviderError):
            await breaker.call(_fail_with, ProviderError("x", reason=reason, retryable=True))

    assert breaker.state is CircuitState.OPEN


async def test_an_open_breaker_raises_the_configured_error_and_names_its_target() -> None:
    breaker = CircuitBreaker(failure_threshold=1, name="groq", error_type=ProviderError)
    with pytest.raises(ProviderError):
        await breaker.call(_fail_with, ProviderError("x", reason="server_error", retryable=True))

    with pytest.raises(ProviderError, match="groq") as excinfo:
        await breaker.call(_fail_with, ValueError("never runs"))

    assert excinfo.value.reason == "circuit_open"
    assert excinfo.value.retryable is False  # retrying an open breaker is pointless


async def test_an_open_breaker_does_not_call_the_dependency() -> None:
    breaker = CircuitBreaker(failure_threshold=1, name="test-skip", error_type=ProviderError)
    with pytest.raises(ProviderError):
        await breaker.call(_fail_with, ProviderError("x", reason="timeout", retryable=True))
    calls = 0

    async def dependency() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(ProviderError):
        await breaker.call(dependency)

    assert calls == 0


async def test_breaker_state_and_openings_are_published_as_metrics() -> None:
    name = "test-metrics"
    opens_before = metric_value("medagent_circuit_breaker_opens_total", target=name)
    breaker = CircuitBreaker(failure_threshold=1, name=name, error_type=ProviderError)
    assert metric_value("medagent_circuit_breaker_state", target=name) == 0

    with pytest.raises(ProviderError):
        await breaker.call(_fail_with, ProviderError("x", reason="server_error", retryable=True))

    assert metric_value("medagent_circuit_breaker_state", target=name) == 2
    assert metric_value("medagent_circuit_breaker_opens_total", target=name) - opens_before == 1

    async def ok() -> str:
        return "ok"

    breaker._opened_at = 0.0  # recovery window has long passed
    assert await breaker.call(ok) == "ok"
    assert metric_value("medagent_circuit_breaker_state", target=name) == 0

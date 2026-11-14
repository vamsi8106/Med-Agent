import pytest

from medagent.infra.retry import retry


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures() -> None:
    calls = {"count": 0}

    @retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
    async def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("boom")
        return "ok"

    assert await flaky() == "ok"
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_retry_raises_after_max_attempts() -> None:
    @retry(max_attempts=2, base_delay=0.01, exceptions=(ValueError,))
    async def always_fails() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        await always_fails()


# --- upstream-aware policy ---------------------------------------------------------

from medagent.core.exceptions import MCPError, ProviderError  # noqa: E402
from tests.conftest import FakeClock, metric_value  # noqa: E402


def _flaky(errors: list[Exception]):
    calls = {"count": 0}

    async def func() -> str:
        calls["count"] += 1
        if errors:
            raise errors.pop(0)
        return "ok"

    return func, calls


async def test_a_permanent_error_fails_on_the_first_attempt(clock: FakeClock) -> None:
    """A 404/400 cannot be fixed by trying again; the old policy burned every attempt."""
    func, calls = _flaky([ProviderError("no such model", reason="client_error")] * 3)

    with pytest.raises(ProviderError):
        await retry(max_attempts=3)(func)()

    assert calls["count"] == 1
    assert clock.sleeps == []


async def test_a_retryable_error_is_retried_with_exponential_backoff(clock: FakeClock) -> None:
    func, calls = _flaky([MCPError("down", reason="server_error", retryable=True)] * 5)

    with pytest.raises(MCPError):
        await retry(max_attempts=4, base_delay=1.0, max_delay=8.0)(func)()

    assert calls["count"] == 4
    assert len(clock.sleeps) == 3
    for slept, base in zip(clock.sleeps, (1.0, 2.0, 4.0), strict=True):
        assert base <= slept <= base * 1.1  # backoff plus at most 10% jitter


async def test_a_short_retry_after_is_honoured_instead_of_the_backoff(clock: FakeClock) -> None:
    func, calls = _flaky(
        [ProviderError("slow down", reason="rate_limited", retryable=True, retry_after=2.5)]
    )

    assert await retry(max_attempts=3, base_delay=0.5, max_wait=10)(func)() == "ok"

    assert calls["count"] == 2
    assert 2.5 <= clock.sleeps[0] <= 3.0


async def test_a_retry_after_longer_than_max_wait_fails_fast_without_sleeping(
    clock: FakeClock,
) -> None:
    """Groq's daily cap says "try again in 8m": holding a doctor's request open
    for that helps nobody, so the error is raised at once."""
    func, calls = _flaky(
        [ProviderError("daily cap", reason="rate_limited", retryable=True, retry_after=487.0)]
    )

    with pytest.raises(ProviderError) as excinfo:
        await retry(max_attempts=3, max_wait=10)(func)()

    assert calls["count"] == 1
    assert clock.sleeps == []
    assert excinfo.value.retry_after == 487.0


async def test_the_total_budget_stops_retrying_early(clock: FakeClock) -> None:
    func, calls = _flaky([MCPError("down", reason="timeout", retryable=True)] * 5)

    with pytest.raises(MCPError):
        # attempt 1 fails -> sleep 10 (fits the budget); attempt 2 fails -> the
        # next sleep of 20 would blow the 15s budget, so it stops.
        await retry(max_attempts=5, base_delay=10, max_delay=100, budget=15)(func)()

    assert calls["count"] == 2
    assert len(clock.sleeps) == 1


async def test_an_error_outside_the_upstream_family_is_not_retried_by_default(
    clock: FakeClock,
) -> None:
    func, calls = _flaky([ValueError("a bug")])

    with pytest.raises(ValueError):
        await retry(max_attempts=3)(func)()

    assert calls["count"] == 1


async def test_retries_are_counted_by_target_and_reason(clock: FakeClock) -> None:
    before = metric_value("medagent_retry_attempts_total", target="test-llm", reason="timeout")
    func, _ = _flaky([ProviderError("t", reason="timeout", retryable=True)] * 2)

    await retry(max_attempts=3, target="test-llm")(func)()

    after = metric_value("medagent_retry_attempts_total", target="test-llm", reason="timeout")
    assert after - before == 2

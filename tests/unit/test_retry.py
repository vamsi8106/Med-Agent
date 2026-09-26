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

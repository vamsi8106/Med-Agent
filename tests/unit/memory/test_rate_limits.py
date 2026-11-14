import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from medagent.memory.persistent import PersistentStore
from medagent.memory.rate_limits import PostgresKeyedLimiter


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _limiter(store: PersistentStore, clock: _Clock, **overrides: object) -> PostgresKeyedLimiter:
    options: dict = {"name": "login_user", "rate_per_second": 0.5, "capacity": 3, "now": clock}
    options.update(overrides)
    return PostgresKeyedLimiter(store, **options)


async def test_two_replicas_share_one_budget(pg_dsn: str) -> None:
    clock = _Clock()
    # Two limiters on two pools: two app replicas.
    pod_a, pod_b = PersistentStore(pg_dsn), PersistentStore(pg_dsn)
    a, b = _limiter(pod_a, clock), _limiter(pod_b, clock)

    results = [await a.try_acquire("dr.alpha"), await b.try_acquire("dr.alpha")]
    results.append(await a.try_acquire("dr.alpha"))

    assert results == [None, None, None]
    assert await b.try_acquire("dr.alpha") == pytest.approx(2.0)
    await pod_a.close()
    await pod_b.close()


async def test_refills_over_time(pg_dsn: str) -> None:
    clock = _Clock()
    store = PersistentStore(pg_dsn)
    limiter = _limiter(store, clock, capacity=1)
    assert await limiter.try_acquire("k") is None
    assert await limiter.try_acquire("k") is not None

    clock.now += timedelta(seconds=2)

    assert await limiter.try_acquire("k") is None
    await store.close()


async def test_keys_and_limiter_names_are_independent(pg_dsn: str) -> None:
    clock = _Clock()
    store = PersistentStore(pg_dsn)
    login = _limiter(store, clock, capacity=1)
    runs = _limiter(store, clock, name="agent_runs", capacity=1)

    assert await login.try_acquire("a") is None
    assert await login.try_acquire("a") is not None
    assert await login.try_acquire("b") is None
    assert await runs.try_acquire("a") is None
    await store.close()


async def test_concurrent_requests_never_overspend(pg_dsn: str) -> None:
    clock = _Clock()
    stores = [PersistentStore(pg_dsn) for _ in range(3)]
    limiters = [_limiter(store, clock, capacity=5, rate_per_second=0.001) for store in stores]

    results = await asyncio.gather(*(limiters[i % 3].try_acquire("hot") for i in range(20)))

    assert sum(result is None for result in results) == 5
    for store in stores:
        await store.close()


async def test_refilled_buckets_are_pruned(pg_dsn: str) -> None:
    clock = _Clock()
    store = PersistentStore(pg_dsn)
    limiter = _limiter(store, clock, capacity=1, rate_per_second=1.0, prune_probability=1.0)
    await limiter.try_acquire("old")
    clock.now += timedelta(seconds=10)  # "old" has long refilled

    await limiter.try_acquire("new")

    async with store.connect() as conn:
        keys = [row["key"] for row in await conn.fetch("SELECT key FROM rate_limit_buckets")]
    assert keys == ["login_user:new"]
    await store.close()

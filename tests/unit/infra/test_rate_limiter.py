import pytest

from medagent.infra.rate_limiter import KeyedRateLimiter, RateLimiter, TokenBucket


@pytest.mark.asyncio
async def test_token_bucket_allows_within_capacity() -> None:
    bucket = TokenBucket(rate_per_second=10, capacity=5)
    for _ in range(5):
        await bucket.acquire()


@pytest.mark.asyncio
async def test_rate_limiter_unknown_source_raises() -> None:
    limiter = RateLimiter()
    with pytest.raises(KeyError):
        await limiter.acquire("unknown-source")


@pytest.mark.asyncio
async def test_rate_limiter_registered_source_acquires() -> None:
    limiter = RateLimiter()
    limiter.register("fda", rate_per_second=10, capacity=5)
    await limiter.acquire("fda")


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_keyed_limiter_allows_the_burst_then_says_how_long_to_wait() -> None:
    clock = _Clock()
    limiter = KeyedRateLimiter(rate_per_second=0.5, capacity=3, now=clock)

    assert [limiter.try_acquire("dr.alpha") for _ in range(3)] == [None, None, None]
    assert limiter.try_acquire("dr.alpha") == pytest.approx(2.0)


def test_keyed_limiter_refills_over_time() -> None:
    clock = _Clock()
    limiter = KeyedRateLimiter(rate_per_second=0.5, capacity=1, now=clock)
    assert limiter.try_acquire("k") is None
    assert limiter.try_acquire("k") is not None

    clock.now += 2.0

    assert limiter.try_acquire("k") is None


def test_keyed_limiter_keys_are_independent() -> None:
    limiter = KeyedRateLimiter(rate_per_second=0.1, capacity=1, now=_Clock())
    assert limiter.try_acquire("a") is None
    assert limiter.try_acquire("a") is not None
    assert limiter.try_acquire("b") is None


def test_keyed_limiter_memory_stays_bounded() -> None:
    clock = _Clock()
    limiter = KeyedRateLimiter(rate_per_second=1.0, capacity=1, max_keys=100, now=clock)

    for i in range(1_000):
        limiter.try_acquire(f"random-user-{i}")

    assert len(limiter) <= 100

    # An exhausted key keeps its state across pruning when it is recent.
    limiter.try_acquire("victim")
    assert limiter.try_acquire("victim") is not None

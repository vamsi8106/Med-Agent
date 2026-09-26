import pytest

from medagent.infra.rate_limiter import RateLimiter, TokenBucket


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

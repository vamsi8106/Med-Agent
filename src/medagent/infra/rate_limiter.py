"""Token-bucket rate limiter, one bucket per API source."""

import asyncio
import time


class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: int) -> None:
        self._rate = rate_per_second
        self._capacity = capacity
        self._tokens = float(capacity)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                await asyncio.sleep(deficit / self._rate)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._updated_at = now


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}

    def register(self, source: str, rate_per_second: float, capacity: int) -> None:
        self._buckets[source] = TokenBucket(rate_per_second, capacity)

    async def acquire(self, source: str, tokens: int = 1) -> None:
        bucket = self._buckets.get(source)
        if bucket is None:
            raise KeyError(f"No rate limit bucket registered for source: {source}")
        await bucket.acquire(tokens)

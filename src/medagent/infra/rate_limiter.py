"""Token-bucket rate limiters.

TokenBucket / RateLimiter throttle our outbound calls (one bucket per MCP
server) by waiting. KeyedRateLimiter limits inbound use (login attempts, agent
runs) per key and never waits: it says how long the caller must back off.
"""

import asyncio
import time
from collections.abc import Callable


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


class KeyedRateLimiter:
    """One token bucket per key (a username, a client IP, a user id), created on
    first use. Non-blocking: try_acquire returns None when allowed, otherwise the
    seconds until a token is available. State is per process.

    Buckets that have refilled completely carry no information, so they are
    dropped when the map grows past max_keys -- memory stays bounded even when
    keys are attacker-chosen (random usernames)."""

    def __init__(
        self,
        rate_per_second: float,
        capacity: int,
        *,
        max_keys: int = 10_000,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rate = rate_per_second
        self._capacity = float(capacity)
        self._max_keys = max_keys
        self._now = now
        # key -> (tokens, updated_at)
        self._buckets: dict[str, tuple[float, float]] = {}

    def try_acquire(self, key: str) -> float | None:
        now = self._now()
        tokens = self._tokens(key, now)
        if tokens >= 1:
            self._buckets[key] = (tokens - 1, now)
            self._prune(now)
            return None
        self._buckets[key] = (tokens, now)
        return (1 - tokens) / self._rate

    def _tokens(self, key: str, now: float) -> float:
        stored = self._buckets.get(key)
        if stored is None:
            return self._capacity
        tokens, updated_at = stored
        return min(self._capacity, tokens + (now - updated_at) * self._rate)

    def _prune(self, now: float) -> None:
        if len(self._buckets) <= self._max_keys:
            return
        full = [key for key in self._buckets if self._tokens(key, now) >= self._capacity]
        for key in full:
            del self._buckets[key]
        # Still over: every key is mid-burst. Drop the oldest-updated ones.
        overflow = len(self._buckets) - self._max_keys
        if overflow > 0:
            oldest = sorted(self._buckets, key=lambda k: self._buckets[k][1])[:overflow]
            for key in oldest:
                del self._buckets[key]

    def __len__(self) -> int:
        return len(self._buckets)

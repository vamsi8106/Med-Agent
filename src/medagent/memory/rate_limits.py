"""Abuse limits shared by every app replica (STATE_BACKEND=postgres).

The same token bucket as infra/rate_limiter.py::KeyedRateLimiter, kept in the
rate_limit_buckets table instead of process memory, so N replicas enforce one
limit rather than N. Each acquire is one short transaction that locks the
bucket's row: concurrent requests for the same key, on any replica, queue on
that lock and never over-spend. Keys are prefixed with the limiter's name
(login_user, login_ip, agent_runs) so limiters never share a bucket.

Buckets idle long enough to have refilled carry no information; they are
deleted now and then, so the table stays small even when keys are
attacker-chosen (random usernames).
"""

import random
from collections.abc import Callable
from datetime import UTC, datetime

import asyncpg

from medagent.core.interfaces import BaseKeyedLimiter
from medagent.memory.persistent import PersistentStore

_PRUNE_PROBABILITY = 0.01


class PostgresKeyedLimiter(BaseKeyedLimiter):
    def __init__(
        self,
        store: PersistentStore,
        name: str,
        rate_per_second: float,
        capacity: int,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        prune_probability: float = _PRUNE_PROBABILITY,
    ) -> None:
        self._store = store
        self._name = name
        self._rate = rate_per_second
        self._capacity = float(capacity)
        self._now = now
        self._prune_probability = prune_probability

    async def try_acquire(self, key: str) -> float | None:
        bucket_key = f"{self._name}:{key}"
        now = self._now()
        async with self._store.connect() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO rate_limit_buckets (key, tokens, updated_at) VALUES ($1, $2, $3) "
                "ON CONFLICT (key) DO NOTHING",
                bucket_key,
                self._capacity,
                now,
            )
            row = await conn.fetchrow(
                "SELECT tokens, updated_at FROM rate_limit_buckets WHERE key = $1 FOR UPDATE",
                bucket_key,
            )
            assert row is not None
            elapsed = max(0.0, (now - row["updated_at"]).total_seconds())
            tokens = min(self._capacity, row["tokens"] + elapsed * self._rate)
            allowed = tokens >= 1
            await conn.execute(
                "UPDATE rate_limit_buckets SET tokens = $2, updated_at = $3 WHERE key = $1",
                bucket_key,
                tokens - 1 if allowed else tokens,
                now,
            )
            if random.random() < self._prune_probability:
                await self._prune(conn, now)
        return None if allowed else (1 - tokens) / self._rate

    async def _prune(self, conn: asyncpg.Connection, now: datetime) -> None:
        # A bucket untouched for capacity / rate seconds has refilled completely.
        full_after = self._capacity / self._rate
        await conn.execute(
            "DELETE FROM rate_limit_buckets WHERE starts_with(key, $1) "
            "AND updated_at < $2::timestamptz - make_interval(secs => $3::double precision)",
            f"{self._name}:",
            now,
            full_after,
        )

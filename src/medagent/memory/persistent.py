"""Postgres connection pool management.

Schema is owned by Alembic migrations (see migrations/), not created here —
this module only manages the runtime connection pool that patient_store.py
and auth/store.py issue queries through.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg


class PersistentStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10)
        return self._pool

    async def init_schema(self) -> None:
        """Ensure the connection pool is ready. Schema itself comes from Alembic."""
        await self._get_pool()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[asyncpg.pool.PoolConnectionProxy]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            yield connection

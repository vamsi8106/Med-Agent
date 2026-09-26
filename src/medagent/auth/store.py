"""Postgres-backed user store for authentication."""

import uuid
from datetime import UTC, datetime

import asyncpg

from medagent.core.exceptions import AuthError
from medagent.core.models import User
from medagent.memory.persistent import PersistentStore


class UserStore:
    def __init__(self, store: PersistentStore) -> None:
        self._store = store

    async def create_user(self, username: str, hashed_password: str, role: str = "doctor") -> User:
        user_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        async with self._store.connect() as conn:
            try:
                await conn.execute(
                    "INSERT INTO users (id, username, hashed_password, role, created_at) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    user_id,
                    username,
                    hashed_password,
                    role,
                    now,
                )
            except asyncpg.UniqueViolationError as exc:
                raise AuthError(f"Username already registered: {username}") from exc

        return User(id=user_id, username=username, role=role, created_at=now)

    async def get_by_username(self, username: str) -> tuple[User, str] | None:
        async with self._store.connect() as conn:
            row = await conn.fetchrow("SELECT * FROM users WHERE username = $1", username)

        if row is None:
            return None

        user = User(
            id=row["id"], username=row["username"], role=row["role"], created_at=row["created_at"]
        )
        return user, row["hashed_password"]

    async def count_users(self) -> int:
        async with self._store.connect() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM users")

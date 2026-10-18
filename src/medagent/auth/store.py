"""Postgres-backed user store for authentication."""

import uuid
from datetime import UTC, datetime

import asyncpg

from medagent.core.exceptions import AuthError
from medagent.core.models import User
from medagent.memory.persistent import PersistentStore


def _user(row: asyncpg.Record) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        role=row["role"],
        disabled=row["disabled"],
        token_version=row["token_version"],
        created_at=row["created_at"],
    )


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

        return _user(row), row["hashed_password"]

    async def count_users(self) -> int:
        async with self._store.connect() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM users")

    async def list_users(self) -> list[User]:
        async with self._store.connect() as conn:
            rows = await conn.fetch("SELECT * FROM users ORDER BY created_at, username")
        return [_user(row) for row in rows]

    async def set_disabled(self, username: str, disabled: bool) -> bool:
        """Disabling also revokes every token. False when the user doesn't exist."""
        async with self._store.connect() as conn:
            status = await conn.execute(
                """UPDATE users SET disabled = $1,
                token_version = token_version + CASE WHEN $1 THEN 1 ELSE 0 END
                WHERE username = $2""",
                disabled,
                username,
            )
        return status != "UPDATE 0"

    async def set_password(self, username: str, hashed_password: str) -> bool:
        """New password; every existing token stops working."""
        async with self._store.connect() as conn:
            status = await conn.execute(
                """UPDATE users SET hashed_password = $1, token_version = token_version + 1
                WHERE username = $2""",
                hashed_password,
                username,
            )
        return status != "UPDATE 0"

    async def revoke_tokens(self, username: str) -> bool:
        async with self._store.connect() as conn:
            status = await conn.execute(
                "UPDATE users SET token_version = token_version + 1 WHERE username = $1",
                username,
            )
        return status != "UPDATE 0"

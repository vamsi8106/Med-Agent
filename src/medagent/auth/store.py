"""SQLite-backed user store for authentication."""

import uuid
from datetime import UTC, datetime

from medagent.core.exceptions import AuthError
from medagent.core.models import User
from medagent.memory.persistent import PersistentStore


class UserStore:
    def __init__(self, store: PersistentStore) -> None:
        self._store = store

    async def create_user(self, username: str, hashed_password: str, role: str = "doctor") -> User:
        async with self._store.connect() as db:
            existing = await (
                await db.execute("SELECT id FROM users WHERE username = ?", (username,))
            ).fetchone()
            if existing is not None:
                raise AuthError(f"Username already registered: {username}")

            user_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            await db.execute(
                "INSERT INTO users (id, username, hashed_password, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, hashed_password, role, now),
            )
            await db.commit()

        return User(id=user_id, username=username, role=role, created_at=now)

    async def get_by_username(self, username: str) -> tuple[User, str] | None:
        async with self._store.connect() as db:
            row = await (
                await db.execute("SELECT * FROM users WHERE username = ?", (username,))
            ).fetchone()

        if row is None:
            return None

        user = User(
            id=row["id"], username=row["username"], role=row["role"], created_at=row["created_at"]
        )
        return user, row["hashed_password"]

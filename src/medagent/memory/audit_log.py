"""Durable audit trail: who accessed or changed which patient record, and when.

Separate from infra/logging.py's structured app logs, which are operational
(request timing, errors) and not meant to be a permanent compliance record.
Rows here are never deleted or rotated by the app.
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from medagent.core.models import User
from medagent.memory.persistent import PersistentStore


class AuditLogStore:
    def __init__(self, store: PersistentStore) -> None:
        self._store = store

    async def record(
        self,
        user: User | None,
        patient_id: str | None,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        async with self._store.connect() as conn:
            await conn.execute(
                """INSERT INTO audit_log
                (id, patient_id, user_id, username, action, details, created_at)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)""",
                str(uuid.uuid4()),
                patient_id,
                user.id if user else None,
                user.username if user else None,
                action,
                json.dumps(details or {}),
                datetime.now(UTC),
            )

    async def for_patient(self, patient_id: str) -> list[dict[str, Any]]:
        async with self._store.connect() as conn:
            rows = await conn.fetch(
                "SELECT * FROM audit_log WHERE patient_id = $1 ORDER BY created_at DESC",
                patient_id,
            )
        return [
            {
                "id": row["id"],
                "patient_id": row["patient_id"],
                "user_id": row["user_id"],
                "username": row["username"],
                "action": row["action"],
                "details": json.loads(row["details"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

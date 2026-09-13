"""In-process patient context CRUD.

Phase 2 scope only: an in-memory store so agents have a PatientContext seam to
call today. Phase 4 adds durable SQLite-backed storage in memory/patient_store.py;
tools/ may never import memory/ (see dependency rules in AGENTS.md), so this tool
stays a thin, swappable cache rather than the system of record.
"""

from datetime import UTC, datetime

from medagent.core.exceptions import ToolError
from medagent.core.models import PatientContext
from medagent.tools.base import BaseTool, ToolResult


class PatientContextTool(BaseTool):
    name = "patient_context"
    description = "Create, read, and update in-memory patient context records."

    def __init__(self) -> None:
        self._store: dict[str, PatientContext] = {}

    async def run(self, action: str, patient_id: str, **kwargs: object) -> ToolResult:
        try:
            if action == "get":
                data = self._get(patient_id)
            elif action == "save":
                context = kwargs.get("context")
                if not isinstance(context, PatientContext):
                    raise ToolError("save action requires a 'context' PatientContext argument")
                data = self._save(context)
            else:
                raise ToolError(f"Unknown patient_context action: {action}")
        except ToolError as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))
        return ToolResult(tool_name=self.name, success=True, data=data)

    def _get(self, patient_id: str) -> PatientContext | None:
        return self._store.get(patient_id)

    def _save(self, context: PatientContext) -> PatientContext:
        now = datetime.now(UTC)
        if context.id not in self._store:
            context.created_at = context.created_at or now
        context.updated_at = now
        self._store[context.id] = context
        return context

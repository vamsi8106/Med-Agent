"""Checkpointer for paused follow-up drafts.

In-memory on purpose: a checkpoint holds the full PatientContext and the draft
report, i.e. patient data. Keeping it in process memory matches where the draft
lived before (a local variable in the WebSocket handler) and stays outside the
per-doctor-isolated Postgres tables and their audit trail. The cost is that a
server restart loses pending drafts, and that it works for a single replica
only -- a persistent checkpointer would need its own retention and doctor_id
isolation for that data.

LangGraph deserializes checkpoint state through an allowlist of types and warns
that unregistered ones "will be blocked in a future version", so every type
that can appear in the follow-up/assessment state is registered explicitly.
"""

from datetime import UTC, datetime, timedelta

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from medagent.infra.logging import get_logger

logger = get_logger(__name__)

_CHECKPOINT_STATE_TYPES: list[tuple[str, str]] = [
    ("medagent.core.models", "PatientContext"),
    ("medagent.core.models", "Medication"),
    ("medagent.core.models", "LabResult"),
    ("medagent.core.models", "Visit"),
    ("medagent.core.models", "RecordChanges"),
    ("medagent.core.models", "MedicationChange"),
    ("medagent.core.models", "LabChange"),
    ("medagent.core.models", "AgentResult"),
    ("medagent.core.models", "AgentFailure"),
    ("medagent.core.models", "ClinicalEvidence"),
    ("medagent.core.models", "DrugInteraction"),
    ("medagent.core.types", "AgentRole"),
    ("medagent.core.types", "InteractionSeverity"),
    ("medagent.core.types", "EvidenceGrade"),
    ("medagent.infra.agent_run", "AgentRunTracker"),
    ("medagent.infra.agent_run", "AgentStepRecord"),
]


def build_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_STATE_TYPES)


def build_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=build_serializer())


async def purge_expired_drafts(
    saver: BaseCheckpointSaver, ttl: timedelta, now: datetime | None = None
) -> int:
    """Deletes every thread whose latest checkpoint is older than `ttl`, so
    drafts abandoned by a dropped connection don't hold patient data in memory
    indefinitely. Returns how many threads were removed."""
    current = now or datetime.now(UTC)
    latest: dict[str, datetime] = {}
    async for checkpoint_tuple in saver.alist(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        written = datetime.fromisoformat(checkpoint_tuple.checkpoint["ts"])
        if thread_id not in latest or written > latest[thread_id]:
            latest[thread_id] = written

    expired = [thread for thread, written in latest.items() if current - written > ttl]
    for thread_id in expired:
        await saver.adelete_thread(thread_id)
    if expired:
        logger.info("approval_drafts_purged", count=len(expired))
    return len(expired)

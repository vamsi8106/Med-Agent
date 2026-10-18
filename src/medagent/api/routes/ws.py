"""Interactive chat over WebSocket with a human-in-the-loop approval gate."""

import asyncio
import json
import uuid
from datetime import timedelta

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from medagent.api.errors import doctor_scope, error_payload, require_doctor_id
from medagent.api.schemas import MAX_MESSAGE_CHARS
from medagent.api.state import AppState, enforce_limit, get_state
from medagent.auth.dependencies import user_for_ws_token
from medagent.core.exceptions import MedAgentError
from medagent.core.models import User
from medagent.infra.logging import get_logger
from medagent.infra.tracing import get_tracer
from medagent.workflows.checkpointing import purge_expired_drafts
from medagent.workflows.followup import (
    ApprovalOutcome,
    discard_draft,
    draft_followup,
    pending_draft,
    resolve_followup,
)

logger = get_logger(__name__)
tracer = get_tracer(__name__)

router = APIRouter()

AUTH_TIMEOUT_SECONDS = 10.0
_POLICY_VIOLATION = 1008


async def _authenticate(websocket: WebSocket, state: AppState) -> User | None:
    """The token comes in the first message, {"type": "auth", "token": "..."},
    not in the URL: URLs end up in proxy and load-balancer logs."""
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS)
        frame = json.loads(raw)
    except (TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        return None
    if not isinstance(frame, dict) or frame.get("type") != "auth":
        return None
    token = frame.get("token")
    if not isinstance(token, str) or not token:
        return None
    return await user_for_ws_token(websocket, token)


@router.websocket("/ws/{patient_id}")
async def chat(websocket: WebSocket, patient_id: str, state: AppState = Depends(get_state)) -> None:
    """Interactive chat with a human-in-the-loop approval gate.

    Each doctor message drafts a report but does NOT save it. While it
    runs the server streams {"type": "progress", "step": ..., "status": ...}
    frames, then sends {"type": "pending_approval", "report": ...} and waits
    for a decision: "approve" saves it verbatim, "reject" discards it, and
    any other text is treated as the doctor's edited version of the report
    and saved in its place.

    A draft survives a dropped connection. Reconnect with ?resume=1 to get
    it back ("resumed": true) and decide; connecting without it discards
    any stale draft, so a fresh question is never mistaken for an edit.

    The first message must authenticate: {"type": "auth", "token": "<JWT>"},
    within AUTH_TIMEOUT_SECONDS; otherwise the socket is closed with 1008.
    """
    await websocket.accept()
    user = await _authenticate(websocket, state)
    if user is None:
        await websocket.close(code=_POLICY_VIOLATION, reason="Not authenticated")
        return

    graph = state.followup_graph
    thread_id = f"{user.id}:{patient_id}"
    # The HTTP middleware doesn't see WebSockets: correlate this session's
    # log lines with an id of its own.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=uuid.uuid4().hex)
    try:
        await purge_expired_drafts(
            state.checkpointer, timedelta(minutes=state.settings.approval_draft_ttl_minutes)
        )
        stale = await pending_draft(graph, thread_id)
        if stale is not None:
            if websocket.query_params.get("resume") in ("1", "true"):
                await state.audit_log.record(user, patient_id, "report_draft_resumed")
                await websocket.send_json(
                    {"type": "pending_approval", "report": stale.report, "resumed": True}
                )
                await _await_decision(websocket, state, user, patient_id, thread_id)
            else:
                await discard_draft(graph, thread_id)
                await state.audit_log.record(user, patient_id, "report_draft_discarded")

        while True:
            message = await websocket.receive_text()
            if not message.strip() or len(message) > MAX_MESSAGE_CHARS:
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": f"Message must be 1-{MAX_MESSAGE_CHARS} characters",
                    }
                )
                continue
            try:
                enforce_limit(
                    state.agent_runs,
                    user.id,
                    limit="agent_runs",
                    message="Too many agent runs; try again later",
                )
                with tracer.start_as_current_span("ws.draft"):
                    draft = await draft_followup(
                        graph,
                        patient_id=patient_id,
                        message=message,
                        doctor_id=doctor_scope(user),
                        thread_id=thread_id,
                        on_progress=websocket.send_json,
                    )
            except MedAgentError as exc:
                await websocket.send_json({"type": "error", **error_payload(exc)})
                continue

            await state.audit_log.record(user, patient_id, "report_drafted", {"message": message})
            await websocket.send_json({"type": "pending_approval", "report": draft.report})
            await _await_decision(websocket, state, user, patient_id, thread_id)
    except WebSocketDisconnect:
        # A pending draft is deliberately left in place for ?resume=1.
        logger.info("websocket_disconnected")


async def _await_decision(
    websocket: WebSocket, state: AppState, user: User, patient_id: str, thread_id: str
) -> None:
    decision = (await websocket.receive_text()).strip()
    outcome = await resolve_followup(state.followup_graph, thread_id=thread_id, decision=decision)
    await _apply_outcome(websocket, state, user, patient_id, outcome)


async def _apply_outcome(
    websocket: WebSocket,
    state: AppState,
    user: User,
    patient_id: str,
    outcome: ApprovalOutcome,
) -> None:
    if outcome.kind == "rejected":
        await state.audit_log.record(user, patient_id, "report_rejected")
        await websocket.send_json({"type": "rejected"})
        return

    assert outcome.report is not None
    await state.patient_store.save_patient(outcome.context)
    await state.patient_store.save_visit(
        patient_id, require_doctor_id(outcome.context), outcome.message, outcome.report
    )
    if outcome.kind == "approved":
        await state.audit_log.record(user, patient_id, "report_approved")
    else:
        await state.audit_log.record(
            user, patient_id, "report_edited", {"edited_report": outcome.report}
        )
    await websocket.send_json({"type": "saved", "report": outcome.report})

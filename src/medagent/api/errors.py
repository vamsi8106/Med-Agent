"""Error mapping and per-doctor scoping shared by the route modules."""

import math
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from medagent.core.exceptions import MedAgentError, UpstreamError
from medagent.core.models import PatientContext, User


def doctor_scope(user: User) -> str | None:
    """Per-doctor data isolation filter: None means unrestricted (admin)."""
    return None if user.role == "admin" else user.id


def retry_after(exc: MedAgentError) -> float | None:
    """The wait to advertise, only for a 429 (rate limit) that knows it."""
    if exc.status_code != 429:
        return None
    wait: float | None = getattr(exc, "retry_after", None)
    return wait


def error_payload(exc: MedAgentError) -> dict[str, Any]:
    """Client-facing body for an error. An upstream error's raw message can carry
    account ids or internal hostnames, so it is replaced by fixed text; our own
    errors (bad input, unknown patient) are already written for the client."""
    payload: dict[str, Any] = {
        "detail": exc.public_message() if isinstance(exc, UpstreamError) else str(exc)
    }
    wait = retry_after(exc)
    if wait is not None:
        payload["retry_after"] = wait
    return payload


def require_doctor_id(patient: PatientContext) -> str:
    """Every persisted patient has doctor_id stamped at creation (see
    create_patient) -- this should never actually be None in practice,
    but save_visit must never silently write a NULL doctor_id since the
    per-doctor isolation guarantee depends on every row having one."""
    assert patient.doctor_id is not None, f"Patient {patient.id} is missing doctor_id"
    return patient.doctor_id


async def medagent_error_handler(_request: Request, exc: MedAgentError) -> JSONResponse:
    headers = {}
    wait = retry_after(exc)
    if wait is not None:
        headers["Retry-After"] = str(math.ceil(wait))
    return JSONResponse(status_code=exc.status_code, content=error_payload(exc), headers=headers)

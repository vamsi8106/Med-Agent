"""Patient records and the REST agent workflows (assess, follow-up, drug check)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from medagent.api.errors import doctor_scope, require_doctor_id
from medagent.api.schemas import AssessRequest, DrugCheckRequest
from medagent.api.state import AppState, get_state
from medagent.auth.dependencies import get_current_user
from medagent.core.models import PatientContext, User
from medagent.workflows.drug_check import run_drug_check
from medagent.workflows.followup import run_followup
from medagent.workflows.patient_assessment import run_patient_assessment

router = APIRouter()


@router.post("/patients", response_model=PatientContext)
async def create_patient(
    context: PatientContext,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> PatientContext:
    # Server-stamped, never client-supplied: whatever doctor_id the
    # request body carried is overwritten with the creating doctor's own
    # identity, and it's immutable after this (see PatientStore.save_patient).
    context.doctor_id = user.id
    await state.patient_store.save_patient(context)
    await state.audit_log.record(user, context.id, "patient_created")
    return context


@router.get("/patients/{patient_id}", response_model=PatientContext)
async def get_patient(
    patient_id: str,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> PatientContext:
    patient = await state.patient_store.get_patient(patient_id, doctor_scope(user))
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
    await state.audit_log.record(user, patient_id, "patient_viewed")
    return patient


@router.get("/patients/{patient_id}/audit-log")
async def get_patient_audit_log(
    patient_id: str,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> list[dict[str, Any]]:
    # Audit log itself isn't doctor-filtered, so ownership must be
    # checked explicitly here before returning any rows.
    patient = await state.patient_store.get_patient(patient_id, doctor_scope(user))
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
    await state.audit_log.record(user, patient_id, "audit_log_viewed")
    return await state.audit_log.for_patient(patient_id)


@router.post("/patients/{patient_id}/assess")
async def assess_patient(
    patient_id: str,
    body: AssessRequest,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    # REST has no interactive approval channel, so it auto-saves. The
    # human-in-the-loop checkpoint lives in the WebSocket chat (routes/ws.py).
    patient = await state.patient_store.get_patient(patient_id, doctor_scope(user))
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Unknown patient: {patient_id}")
    report = await run_patient_assessment(
        state.triage,
        state.drug_safety,
        state.evidence,
        state.trial_finder,
        state.report,
        patient,
        body.message,
    )
    await state.patient_store.save_patient(patient)
    await state.patient_store.save_visit(
        patient_id, require_doctor_id(patient), body.message, report
    )
    await state.audit_log.record(user, patient_id, "assessment_run", {"message": body.message})
    return {"report": report}


@router.post("/patients/{patient_id}/followup")
async def followup_visit(
    patient_id: str,
    body: AssessRequest,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    report, patient = await run_followup(
        state.triage,
        state.drug_safety,
        state.evidence,
        state.trial_finder,
        state.report,
        state.patient_store,
        patient_id,
        body.message,
        doctor_id=doctor_scope(user),
    )
    await state.patient_store.save_patient(patient)
    await state.patient_store.save_visit(
        patient_id, require_doctor_id(patient), body.message, report
    )
    await state.audit_log.record(user, patient_id, "followup_run", {"message": body.message})
    return {"report": report}


@router.post("/drug-check")
async def drug_check(
    body: DrugCheckRequest,
    user: User = Depends(get_current_user),
    state: AppState = Depends(get_state),
) -> dict[str, str]:
    answer = await run_drug_check(
        state.drug_safety,
        state.patient_store,
        body.patient_id,
        body.new_drug,
        doctor_id=doctor_scope(user),
    )
    await state.audit_log.record(
        user, body.patient_id, "drug_check_run", {"new_drug": body.new_drug}
    )
    return {"answer": answer}

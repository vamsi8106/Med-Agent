"""Postgres-backed CRUD for patient records: patients, medications, lab_results, visits.

Medications and lab results are append-only history. A save reconciles the
incoming record with what is stored: a medication that disappears is marked
stopped (the row stays, with end_date), a dose change closes the old row and
opens a new one, and a lab result is a new row. get_patient returns the
current view -- active medications, the latest result per test -- plus what
changed since the newest visit (see record_changes.py).
"""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg

from medagent.core.exceptions import MemoryError as MedAgentMemoryError
from medagent.core.exceptions import PatientOwnershipError
from medagent.core.interfaces import BaseMemory
from medagent.core.models import LabResult, Medication, PatientContext, Visit
from medagent.memory.persistent import PersistentStore
from medagent.memory.record_changes import (
    compute_changes,
    medication_key,
    record_snapshot,
    regimen,
)

_RECENT_VISITS_LIMIT = 5
_STOPPED = "stopped"


class PatientStore(BaseMemory):
    def __init__(self, store: PersistentStore) -> None:
        self._store = store

    async def get_patient(
        self, patient_id: str, doctor_id: str | None = None
    ) -> PatientContext | None:
        async with self._store.connect() as conn:
            if doctor_id is not None:
                patient_row = await conn.fetchrow(
                    "SELECT * FROM patients WHERE id = $1 AND doctor_id = $2",
                    patient_id,
                    doctor_id,
                )
            else:
                patient_row = await conn.fetchrow(
                    "SELECT * FROM patients WHERE id = $1", patient_id
                )
            if patient_row is None:
                return None
            return await self._load(conn, patient_row)

    async def _load(self, conn: asyncpg.Connection, patient_row: asyncpg.Record) -> PatientContext:
        patient_id = patient_row["id"]
        med_rows = await conn.fetch(
            "SELECT * FROM medications WHERE patient_id = $1 AND status <> $2 ORDER BY created_at",
            patient_id,
            _STOPPED,
        )
        lab_rows = await conn.fetch(
            """SELECT DISTINCT ON (lower(test_name)) * FROM lab_results WHERE patient_id = $1
            ORDER BY lower(test_name), collected_at DESC""",
            patient_id,
        )
        visit_rows = await conn.fetch(
            """SELECT * FROM visits WHERE patient_id = $1
            ORDER BY visit_date DESC, created_at DESC LIMIT $2""",
            patient_id,
            _RECENT_VISITS_LIMIT,
        )

        context = PatientContext(
            id=patient_id,
            name=patient_row["name"],
            age=patient_row["age"],
            sex=patient_row["sex"],
            doctor_id=patient_row["doctor_id"],
            weight_kg=patient_row["weight_kg"],
            height_cm=patient_row["height_cm"],
            conditions=json.loads(patient_row["conditions"]),
            allergies=json.loads(patient_row["allergies"]),
            medications=[
                Medication(
                    id=row["id"],
                    name=row["name"],
                    brand_name=row["brand_name"],
                    dose=row["dose"],
                    frequency=row["frequency"],
                    route=row["route"],
                    start_date=row["start_date"],
                    end_date=row["end_date"],
                    status=row["status"],
                )
                for row in med_rows
            ],
            lab_results=[
                LabResult(
                    id=row["id"],
                    test_name=row["test_name"],
                    value=row["value"],
                    unit=row["unit"],
                    reference_low=row["reference_low"],
                    reference_high=row["reference_high"],
                    is_abnormal=row["is_abnormal"],
                    collected_at=row["collected_at"],
                )
                for row in lab_rows
            ],
            visits=[
                Visit(
                    id=row["id"],
                    visit_date=row["visit_date"],
                    chief_complaint=row["chief_complaint"],
                    assessment=row["assessment"],
                    plan=row["plan"],
                    created_at=row["created_at"],
                )
                for row in visit_rows
            ],
            created_at=patient_row["created_at"],
            updated_at=patient_row["updated_at"],
        )
        # Visits saved before snapshots existed have none: no baseline, no diff.
        if visit_rows and visit_rows[0]["record_snapshot"] is not None:
            context.changes_since_last_visit = compute_changes(
                json.loads(visit_rows[0]["record_snapshot"]),
                context,
                since=visit_rows[0]["visit_date"],
            )
        return context

    async def save_patient(self, context: PatientContext) -> None:
        if not context.id:
            raise MedAgentMemoryError("PatientContext.id is required to save a patient record")
        if not context.doctor_id:
            raise MedAgentMemoryError(
                "PatientContext.doctor_id is required to save a patient record"
            )

        now = datetime.now(UTC)
        async with self._store.connect() as conn, conn.transaction():
            existing = await conn.fetchrow(
                "SELECT id, doctor_id, created_at FROM patients WHERE id = $1 FOR UPDATE",
                context.id,
            )
            if existing is not None and existing["doctor_id"] != context.doctor_id:
                # Another doctor's patient. Refused outright: the old UPDATE path
                # let POST /patients overwrite their record under the same id.
                raise PatientOwnershipError()

            if existing is None:
                # doctor_id is set here only -- ownership is immutable after
                # creation. POST /patients server-stamps it from the JWT; every
                # other caller re-saves a context from the doctor-scoped
                # get_patient, which carries the owner already.
                await conn.execute(
                    """INSERT INTO patients
                    (id, name, age, sex, doctor_id, weight_kg, height_cm, conditions,
                     allergies, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11)""",
                    context.id,
                    context.name,
                    context.age,
                    context.sex,
                    context.doctor_id,
                    context.weight_kg,
                    context.height_cm,
                    json.dumps(context.conditions),
                    json.dumps(context.allergies),
                    context.created_at or now,
                    now,
                )
            else:
                await conn.execute(
                    """UPDATE patients SET name = $1, age = $2, sex = $3, weight_kg = $4,
                    height_cm = $5, conditions = $6::jsonb, allergies = $7::jsonb, updated_at = $8
                    WHERE id = $9""",
                    context.name,
                    context.age,
                    context.sex,
                    context.weight_kg,
                    context.height_cm,
                    json.dumps(context.conditions),
                    json.dumps(context.allergies),
                    now,
                    context.id,
                )

            await self._reconcile_medications(conn, context, now)
            await self._append_lab_results(conn, context)

    async def _reconcile_medications(
        self, conn: asyncpg.Connection, context: PatientContext, now: datetime
    ) -> None:
        """Brings the active medications in line with context.medications without
        deleting anything. A re-save of an unchanged record writes nothing."""
        active_rows = await conn.fetch(
            "SELECT * FROM medications WHERE patient_id = $1 AND status <> $2",
            context.id,
            _STOPPED,
        )
        active: dict[str, dict[str, Any]] = {
            medication_key(row["name"]): dict(row) for row in active_rows
        }
        incoming = {medication_key(med.name): med for med in context.medications}

        for key, row in active.items():
            med = incoming.get(key)
            if med is None or med.status == _STOPPED or regimen(med) != regimen(row):
                await conn.execute(
                    "UPDATE medications SET status = $1, end_date = $2 WHERE id = $3",
                    _STOPPED,
                    med.end_date if med is not None and med.end_date else now,
                    row["id"],
                )

        for key, med in incoming.items():
            if med.status == _STOPPED:
                continue
            current = active.get(key)
            if current is not None and regimen(current) == regimen(med):
                continue
            # A regimen change starts now; a new medication keeps a given start.
            # Always a fresh id: a re-saved context carries the old row's id.
            await conn.execute(
                """INSERT INTO medications
                (id, patient_id, doctor_id, name, brand_name, dose, frequency, route,
                 start_date, end_date, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                str(uuid.uuid4()),
                context.id,
                context.doctor_id,
                med.name,
                med.brand_name,
                med.dose,
                med.frequency,
                med.route,
                now if current is not None else (med.start_date or now),
                None,
                med.status,
                now,
            )

    async def _append_lab_results(self, conn: asyncpg.Connection, context: PatientContext) -> None:
        for lab in context.lab_results:
            await conn.execute(
                """INSERT INTO lab_results
                (id, patient_id, doctor_id, test_name, value, unit, reference_low,
                 reference_high, is_abnormal, collected_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (patient_id, test_name, collected_at) DO NOTHING""",
                str(uuid.uuid4()),
                context.id,
                context.doctor_id,
                lab.test_name,
                lab.value,
                lab.unit,
                lab.reference_low,
                lab.reference_high,
                lab.is_abnormal,
                lab.collected_at,
            )

    async def save_visit(
        self, patient_id: str, doctor_id: str, chief_complaint: str, assessment: str
    ) -> None:
        """Appends a visit record with a snapshot of the record as it stands, the
        baseline the next visit's changes are computed against. Insert-only --
        visits are never updated or deleted."""
        now = datetime.now(UTC)
        async with self._store.connect() as conn:
            patient_row = await conn.fetchrow("SELECT * FROM patients WHERE id = $1", patient_id)
            snapshot = (
                record_snapshot(await self._load(conn, patient_row))
                if patient_row is not None
                else None
            )
            await conn.execute(
                """INSERT INTO visits
                (id, patient_id, doctor_id, visit_date, chief_complaint, assessment,
                 record_snapshot, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)""",
                str(uuid.uuid4()),
                patient_id,
                doctor_id,
                now,
                chief_complaint,
                assessment,
                json.dumps(snapshot) if snapshot is not None else None,
                now,
            )

"""Postgres-backed CRUD for patient records: patients, medications, lab_results."""

import json
import uuid
from datetime import UTC, datetime

from medagent.core.exceptions import MemoryError as MedAgentMemoryError
from medagent.core.interfaces import BaseMemory
from medagent.core.models import LabResult, Medication, PatientContext
from medagent.memory.persistent import PersistentStore


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

            med_rows = await conn.fetch(
                "SELECT * FROM medications WHERE patient_id = $1", patient_id
            )
            lab_rows = await conn.fetch(
                "SELECT * FROM lab_results WHERE patient_id = $1", patient_id
            )

        return PatientContext(
            id=patient_row["id"],
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
            created_at=patient_row["created_at"],
            updated_at=patient_row["updated_at"],
        )

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
                "SELECT id, created_at FROM patients WHERE id = $1", context.id
            )
            created_at = (
                existing["created_at"] if existing is not None else (context.created_at or now)
            )

            if existing is None:
                # doctor_id is set here only -- the UPDATE branch below never
                # touches it, so ownership is immutable after creation. Every
                # caller that reaches this branch with an untrusted
                # context.doctor_id is app.py's POST /patients, which
                # server-stamps it from the JWT before calling save_patient;
                # every other caller re-saves a context obtained via the
                # already doctor-scoped get_patient, so it already carries
                # the correct, previously-verified value.
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
                    created_at,
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

            await conn.execute("DELETE FROM medications WHERE patient_id = $1", context.id)
            for med in context.medications:
                await conn.execute(
                    """INSERT INTO medications
                    (id, patient_id, doctor_id, name, brand_name, dose, frequency, route,
                     start_date, end_date, status, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                    med.id or str(uuid.uuid4()),
                    context.id,
                    context.doctor_id,
                    med.name,
                    med.brand_name,
                    med.dose,
                    med.frequency,
                    med.route,
                    med.start_date,
                    med.end_date,
                    med.status,
                    now,
                )

            await conn.execute("DELETE FROM lab_results WHERE patient_id = $1", context.id)
            for lab in context.lab_results:
                await conn.execute(
                    """INSERT INTO lab_results
                    (id, patient_id, doctor_id, test_name, value, unit, reference_low,
                     reference_high, is_abnormal, collected_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                    lab.id or str(uuid.uuid4()),
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

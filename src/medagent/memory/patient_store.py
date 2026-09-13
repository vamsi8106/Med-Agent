"""SQLite-backed CRUD for patient records: patients, medications, lab_results."""

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

    async def get_patient(self, patient_id: str) -> PatientContext | None:
        async with self._store.connect() as db:
            patient_row = await (
                await db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            ).fetchone()
            if patient_row is None:
                return None

            med_rows = await (
                await db.execute("SELECT * FROM medications WHERE patient_id = ?", (patient_id,))
            ).fetchall()
            lab_rows = await (
                await db.execute("SELECT * FROM lab_results WHERE patient_id = ?", (patient_id,))
            ).fetchall()

        return PatientContext(
            id=patient_row["id"],
            name=patient_row["name"],
            age=patient_row["age"],
            sex=patient_row["sex"],
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
                    is_abnormal=bool(row["is_abnormal"]),
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

        now = datetime.now(UTC).isoformat()
        async with self._store.connect() as db:
            existing = await (
                await db.execute("SELECT id FROM patients WHERE id = ?", (context.id,))
            ).fetchone()
            created_at = (context.created_at or now) if existing is None else None

            if existing is None:
                await db.execute(
                    """INSERT INTO patients
                    (id, name, age, sex, weight_kg, height_cm, conditions, allergies,
                     created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        context.id,
                        context.name,
                        context.age,
                        context.sex,
                        context.weight_kg,
                        context.height_cm,
                        json.dumps(context.conditions),
                        json.dumps(context.allergies),
                        str(created_at),
                        now,
                    ),
                )
            else:
                await db.execute(
                    """UPDATE patients SET name = ?, age = ?, sex = ?, weight_kg = ?,
                    height_cm = ?, conditions = ?, allergies = ?, updated_at = ?
                    WHERE id = ?""",
                    (
                        context.name,
                        context.age,
                        context.sex,
                        context.weight_kg,
                        context.height_cm,
                        json.dumps(context.conditions),
                        json.dumps(context.allergies),
                        now,
                        context.id,
                    ),
                )

            await db.execute("DELETE FROM medications WHERE patient_id = ?", (context.id,))
            for med in context.medications:
                await db.execute(
                    """INSERT INTO medications
                    (id, patient_id, name, brand_name, dose, frequency, route,
                     start_date, end_date, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        med.id or str(uuid.uuid4()),
                        context.id,
                        med.name,
                        med.brand_name,
                        med.dose,
                        med.frequency,
                        med.route,
                        med.start_date.isoformat() if med.start_date else None,
                        med.end_date.isoformat() if med.end_date else None,
                        med.status,
                        now,
                    ),
                )

            await db.execute("DELETE FROM lab_results WHERE patient_id = ?", (context.id,))
            for lab in context.lab_results:
                await db.execute(
                    """INSERT INTO lab_results
                    (id, patient_id, test_name, value, unit, reference_low,
                     reference_high, is_abnormal, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        lab.id or str(uuid.uuid4()),
                        context.id,
                        lab.test_name,
                        lab.value,
                        lab.unit,
                        lab.reference_low,
                        lab.reference_high,
                        int(lab.is_abnormal),
                        lab.collected_at.isoformat(),
                    ),
                )

            await db.commit()

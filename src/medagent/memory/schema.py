"""Canonical schema DDL, shared by the Alembic migration and the test suite.

Single source of truth so tests exercise the exact same schema production
gets via `alembic upgrade head` — never a hand-copied approximation of it.
"""

CREATE_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS patients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        sex TEXT NOT NULL,
        weight_kg DOUBLE PRECISION,
        height_cm DOUBLE PRECISION,
        conditions JSONB NOT NULL DEFAULT '[]'::jsonb,
        allergies JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS medications (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id),
        name TEXT NOT NULL,
        brand_name TEXT,
        dose TEXT,
        frequency TEXT,
        route TEXT,
        start_date TIMESTAMPTZ,
        end_date TIMESTAMPTZ,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_medications_patient_id ON medications(patient_id)",
    """
    CREATE TABLE IF NOT EXISTS lab_results (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id),
        test_name TEXT NOT NULL,
        value DOUBLE PRECISION NOT NULL,
        unit TEXT NOT NULL,
        reference_low DOUBLE PRECISION,
        reference_high DOUBLE PRECISION,
        is_abnormal BOOLEAN NOT NULL DEFAULT FALSE,
        collected_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_lab_results_patient_id ON lab_results(patient_id)",
    """
    CREATE TABLE IF NOT EXISTS visits (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id),
        visit_date TIMESTAMPTZ NOT NULL,
        chief_complaint TEXT,
        assessment TEXT,
        plan TEXT,
        agent_session_id TEXT,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_visits_patient_id ON visits(patient_id)",
    """
    CREATE TABLE IF NOT EXISTS interactions_log (
        id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(id),
        drug_a TEXT NOT NULL,
        drug_b TEXT NOT NULL,
        severity TEXT NOT NULL,
        description TEXT NOT NULL,
        source TEXT,
        checked_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_interactions_log_patient_id ON interactions_log(patient_id)",
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'doctor',
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
]

DROP_STATEMENTS: list[str] = [
    "DROP TABLE IF EXISTS users",
    "DROP TABLE IF EXISTS interactions_log",
    "DROP TABLE IF EXISTS visits",
    "DROP TABLE IF EXISTS lab_results",
    "DROP TABLE IF EXISTS medications",
    "DROP TABLE IF EXISTS patients",
]

# Migration 0002: audit trail of who accessed/modified which patient record.
# A separate statement group (not folded into CREATE_STATEMENTS above) since
# it belongs to its own, later migration -- production schema changes are
# additive migrations, never edits to an already-applied one.
#
# patient_id/user_id are deliberately NOT foreign keys: a "report_drafted"
# event can be recorded before a brand-new patient's row is saved, and an
# audit trail must remain readable even after a patient or user record is
# later deleted.
AUDIT_LOG_CREATE_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id TEXT PRIMARY KEY,
        patient_id TEXT,
        user_id TEXT,
        username TEXT,
        action TEXT NOT NULL,
        details JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_audit_log_patient_id ON audit_log(patient_id)",
    "CREATE INDEX IF NOT EXISTS ix_audit_log_created_at ON audit_log(created_at)",
]

AUDIT_LOG_DROP_STATEMENTS: list[str] = [
    "DROP TABLE IF EXISTS audit_log",
]

# Migration 0003: per-doctor data isolation. Nullable at the DB level (a
# NOT NULL column would break `alembic upgrade head` against an already-
# populated patients table with no backfill) -- required-ness is enforced at
# the application layer instead, in PatientStore.save_patient.
ADD_DOCTOR_ID_STATEMENTS: list[str] = [
    "ALTER TABLE patients ADD COLUMN IF NOT EXISTS doctor_id TEXT",
    "CREATE INDEX IF NOT EXISTS ix_patients_doctor_id ON patients(doctor_id)",
    "ALTER TABLE medications ADD COLUMN IF NOT EXISTS doctor_id TEXT",
    "CREATE INDEX IF NOT EXISTS ix_medications_doctor_id ON medications(doctor_id)",
    "ALTER TABLE lab_results ADD COLUMN IF NOT EXISTS doctor_id TEXT",
    "CREATE INDEX IF NOT EXISTS ix_lab_results_doctor_id ON lab_results(doctor_id)",
    "ALTER TABLE visits ADD COLUMN IF NOT EXISTS doctor_id TEXT",
    "CREATE INDEX IF NOT EXISTS ix_visits_doctor_id ON visits(doctor_id)",
    "ALTER TABLE interactions_log ADD COLUMN IF NOT EXISTS doctor_id TEXT",
    "CREATE INDEX IF NOT EXISTS ix_interactions_log_doctor_id ON interactions_log(doctor_id)",
]

REMOVE_DOCTOR_ID_STATEMENTS: list[str] = [
    "DROP INDEX IF EXISTS ix_interactions_log_doctor_id",
    "ALTER TABLE interactions_log DROP COLUMN IF EXISTS doctor_id",
    "DROP INDEX IF EXISTS ix_visits_doctor_id",
    "ALTER TABLE visits DROP COLUMN IF EXISTS doctor_id",
    "DROP INDEX IF EXISTS ix_lab_results_doctor_id",
    "ALTER TABLE lab_results DROP COLUMN IF EXISTS doctor_id",
    "DROP INDEX IF EXISTS ix_medications_doctor_id",
    "ALTER TABLE medications DROP COLUMN IF EXISTS doctor_id",
    "DROP INDEX IF EXISTS ix_patients_doctor_id",
    "ALTER TABLE patients DROP COLUMN IF EXISTS doctor_id",
]

TABLES_IN_DEPENDENCY_ORDER: list[str] = [
    "audit_log",
    "interactions_log",
    "visits",
    "lab_results",
    "medications",
    "patients",
    "users",
]

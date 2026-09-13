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

TABLES_IN_DEPENDENCY_ORDER: list[str] = [
    "interactions_log",
    "visits",
    "lab_results",
    "medications",
    "patients",
    "users",
]

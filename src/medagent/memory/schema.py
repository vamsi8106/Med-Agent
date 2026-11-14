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

# Migration 0004: record history. Medications and labs become append-only
# (PatientStore reconciles instead of delete+insert), so a re-save must not
# duplicate a lab result: one row per (patient, test, collection time). Each
# visit stores a snapshot of the record, the baseline for "what changed".
ADD_RECORD_HISTORY_STATEMENTS: list[str] = [
    "ALTER TABLE visits ADD COLUMN IF NOT EXISTS record_snapshot JSONB",
    """
    DELETE FROM lab_results a USING lab_results b
    WHERE a.patient_id = b.patient_id AND a.test_name = b.test_name
      AND a.collected_at = b.collected_at AND a.id > b.id
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_lab_results_patient_test_time
    ON lab_results(patient_id, test_name, collected_at)
    """,
]

REMOVE_RECORD_HISTORY_STATEMENTS: list[str] = [
    "DROP INDEX IF EXISTS ux_lab_results_patient_test_time",
    "ALTER TABLE visits DROP COLUMN IF EXISTS record_snapshot",
]

# Migration 0005: user management. `disabled` blocks login and every token;
# bumping `token_version` revokes all tokens issued before (each token carries
# the version it was issued with, see auth/security.py).
ADD_USER_MANAGEMENT_STATEMENTS: list[str] = [
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS disabled BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0",
]

REMOVE_USER_MANAGEMENT_STATEMENTS: list[str] = [
    "ALTER TABLE users DROP COLUMN IF EXISTS token_version",
    "ALTER TABLE users DROP COLUMN IF EXISTS disabled",
]

# Migration 0006: shared state, so several app replicas can serve one
# deployment (STATE_BACKEND=postgres).
#
# rate_limit_buckets backs the abuse limits (memory/rate_limits.py): one token
# bucket per limiter name + key, updated under a row lock.
#
# The checkpoint_* tables are LangGraph's AsyncPostgresSaver schema, holding
# WebSocket approval drafts (patient data, purged after APPROVAL_DRAFT_TTL_MINUTES
# by memory/drafts.py). This is a frozen copy of the saver's own MIGRATIONS
# list (langgraph-checkpoint-postgres 3.2.1) so the schema stays Alembic-owned
# and setup() never runs at request time. CREATE INDEX drops CONCURRENTLY:
# Alembic runs inside a transaction, and these tables start empty.
# checkpoint_migrations records every version, which is how the saver knows
# its schema is current; tests/unit/memory/test_drafts.py fails if a package
# upgrade adds a migration this copy lacks.
CHECKPOINT_SAVER_MIGRATIONS: list[str] = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
    v INTEGER PRIMARY KEY
)""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
)""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
)""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
)""",
    "ALTER TABLE checkpoint_blobs ALTER COLUMN blob DROP not null",
    "SELECT 1",
    "CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id)",
    "CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id)",
    "CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id)",
    "ALTER TABLE checkpoint_writes ADD COLUMN IF NOT EXISTS task_path TEXT NOT NULL DEFAULT ''",
]

ADD_SHARED_STATE_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS rate_limit_buckets (
        key TEXT PRIMARY KEY,
        tokens DOUBLE PRECISION NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_rate_limit_buckets_updated_at ON rate_limit_buckets(updated_at)",
    *CHECKPOINT_SAVER_MIGRATIONS,
    f"""INSERT INTO checkpoint_migrations (v)
    SELECT generate_series(0, {len(CHECKPOINT_SAVER_MIGRATIONS) - 1}) ON CONFLICT DO NOTHING""",
]

REMOVE_SHARED_STATE_STATEMENTS: list[str] = [
    "DROP TABLE IF EXISTS checkpoint_writes",
    "DROP TABLE IF EXISTS checkpoint_blobs",
    "DROP TABLE IF EXISTS checkpoints",
    "DROP TABLE IF EXISTS checkpoint_migrations",
    "DROP TABLE IF EXISTS rate_limit_buckets",
]

TABLES_IN_DEPENDENCY_ORDER: list[str] = [
    "checkpoint_writes",
    "checkpoint_blobs",
    "checkpoints",
    "rate_limit_buckets",
    "audit_log",
    "interactions_log",
    "visits",
    "lab_results",
    "medications",
    "patients",
    "users",
]

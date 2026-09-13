"""SQLite schema and connection management for patient records."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    weight_kg REAL,
    height_cm REAL,
    conditions TEXT NOT NULL DEFAULT '[]',
    allergies TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medications (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    name TEXT NOT NULL,
    brand_name TEXT,
    dose TEXT,
    frequency TEXT,
    route TEXT,
    start_date TEXT,
    end_date TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lab_results (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    test_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    reference_low REAL,
    reference_high REAL,
    is_abnormal INTEGER NOT NULL DEFAULT 0,
    collected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS visits (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    visit_date TEXT NOT NULL,
    chief_complaint TEXT,
    assessment TEXT,
    plan TEXT,
    agent_session_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interactions_log (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    drug_a TEXT NOT NULL,
    drug_b TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    source TEXT,
    checked_at TEXT NOT NULL
);
"""


class PersistentStore:
    def __init__(self, database_path: str) -> None:
        self._database_path = database_path

    async def init_schema(self) -> None:
        async with aiosqlite.connect(self._database_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        db = await aiosqlite.connect(self._database_path)
        db.row_factory = aiosqlite.Row
        try:
            yield db
        finally:
            await db.close()

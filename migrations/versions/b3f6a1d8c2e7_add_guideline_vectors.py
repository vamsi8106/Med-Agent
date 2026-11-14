"""add guideline vectors (pgvector replaces ChromaDB)

Revision ID: b3f6a1d8c2e7
Revises: 9e4a2c7b1d63
Create Date: 2026-11-15 10:00:00.000000

On RDS the `vector` extension needs rds_superuser to create; if the app's user
lacks it, a DBA runs `CREATE EXTENSION vector;` once and this migration's
IF NOT EXISTS skips it.
"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import (
    ADD_GUIDELINE_VECTOR_STATEMENTS,
    REMOVE_GUIDELINE_VECTOR_STATEMENTS,
)

# revision identifiers, used by Alembic.
revision: str = "b3f6a1d8c2e7"
down_revision: str | Sequence[str] | None = "9e4a2c7b1d63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in ADD_GUIDELINE_VECTOR_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in REMOVE_GUIDELINE_VECTOR_STATEMENTS:
        op.execute(statement)

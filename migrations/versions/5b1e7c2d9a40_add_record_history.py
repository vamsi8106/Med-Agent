"""add record history

Revision ID: 5b1e7c2d9a40
Revises: 08206ee501f9
Create Date: 2026-09-27 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import (
    ADD_RECORD_HISTORY_STATEMENTS,
    REMOVE_RECORD_HISTORY_STATEMENTS,
)

# revision identifiers, used by Alembic.
revision: str = "5b1e7c2d9a40"
down_revision: str | Sequence[str] | None = "08206ee501f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in ADD_RECORD_HISTORY_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in REMOVE_RECORD_HISTORY_STATEMENTS:
        op.execute(statement)

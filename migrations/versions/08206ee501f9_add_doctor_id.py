"""add doctor_id

Revision ID: 08206ee501f9
Revises: 4ef57aabd914
Create Date: 2026-09-14 17:54:34.353850

"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import ADD_DOCTOR_ID_STATEMENTS, REMOVE_DOCTOR_ID_STATEMENTS

# revision identifiers, used by Alembic.
revision: str = "08206ee501f9"
down_revision: str | Sequence[str] | None = "4ef57aabd914"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in ADD_DOCTOR_ID_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in REMOVE_DOCTOR_ID_STATEMENTS:
        op.execute(statement)

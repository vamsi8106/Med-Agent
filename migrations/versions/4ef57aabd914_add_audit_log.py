"""add audit log

Revision ID: 4ef57aabd914
Revises: d66f7010855b
Create Date: 2026-09-13 17:10:43.907040

"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import AUDIT_LOG_CREATE_STATEMENTS, AUDIT_LOG_DROP_STATEMENTS

# revision identifiers, used by Alembic.
revision: str = "4ef57aabd914"
down_revision: str | Sequence[str] | None = "d66f7010855b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in AUDIT_LOG_CREATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in AUDIT_LOG_DROP_STATEMENTS:
        op.execute(statement)

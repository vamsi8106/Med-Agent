"""initial schema

Revision ID: d66f7010855b
Revises:
Create Date: 2026-09-13 16:45:01.937751

"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import CREATE_STATEMENTS, DROP_STATEMENTS

# revision identifiers, used by Alembic.
revision: str = "d66f7010855b"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in CREATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_STATEMENTS:
        op.execute(statement)

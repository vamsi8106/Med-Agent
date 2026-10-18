"""add user management

Revision ID: 7c3d9e1f2b55
Revises: 5b1e7c2d9a40
Create Date: 2026-10-18 10:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import (
    ADD_USER_MANAGEMENT_STATEMENTS,
    REMOVE_USER_MANAGEMENT_STATEMENTS,
)

# revision identifiers, used by Alembic.
revision: str = "7c3d9e1f2b55"
down_revision: str | Sequence[str] | None = "5b1e7c2d9a40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in ADD_USER_MANAGEMENT_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in REMOVE_USER_MANAGEMENT_STATEMENTS:
        op.execute(statement)

"""add shared state (rate-limit buckets, approval-draft checkpoints)

Revision ID: 9e4a2c7b1d63
Revises: 7c3d9e1f2b55
Create Date: 2026-11-14 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

from medagent.memory.schema import (
    ADD_SHARED_STATE_STATEMENTS,
    REMOVE_SHARED_STATE_STATEMENTS,
)

# revision identifiers, used by Alembic.
revision: str = "9e4a2c7b1d63"
down_revision: str | Sequence[str] | None = "7c3d9e1f2b55"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in ADD_SHARED_STATE_STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for statement in REMOVE_SHARED_STATE_STATEMENTS:
        op.execute(statement)

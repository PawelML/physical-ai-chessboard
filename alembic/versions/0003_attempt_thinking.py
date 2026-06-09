"""record per-attempt model thinking trace

Revision ID: 0003_attempt_thinking
Revises: 0002_game_summary_unfinished
Create Date: 2026-06-09
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003_attempt_thinking"
down_revision: str | None = "0002_game_summary_unfinished"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("attempts", sa.Column("thinking", sa.Text(), nullable=True))
    op.add_column(
        "attempts",
        sa.Column(
            "thinking_used",
            sa.Boolean(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("attempts", "thinking_used")
    op.drop_column("attempts", "thinking")

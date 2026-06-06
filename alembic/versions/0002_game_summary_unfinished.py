"""add unfinished game summary counter

Revision ID: 0002_game_summary_unfinished
Revises: 0001_initial
Create Date: 2026-06-06
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_game_summary_unfinished"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "game_summaries",
        sa.Column("unfinished", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("game_summaries", "unfinished")

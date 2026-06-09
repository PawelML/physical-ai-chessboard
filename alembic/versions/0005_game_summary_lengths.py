"""add average game length to summaries

Revision ID: 0005_game_summary_lengths
Revises: 0004_app_settings
Create Date: 2026-06-09
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_game_summary_lengths"
down_revision: str | None = "0004_app_settings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "game_summaries",
        sa.Column("avg_game_plies", sa.Float(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("game_summaries", "avg_game_plies")

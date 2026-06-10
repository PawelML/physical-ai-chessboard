"""add benchmark summary statistics

Revision ID: 0006_summary_statistics
Revises: 0005_game_summary_lengths
Create Date: 2026-06-10
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_summary_statistics"
down_revision: str | None = "0005_game_summary_lengths"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "game_summaries",
        sa.Column("evaluated_move_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("accuracy_rate", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("illegal_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("malformed_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("illegal_rate_ci_low", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("illegal_rate_ci_high", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("malformed_rate_ci_low", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("malformed_rate_ci_high", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("win_rate", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("win_rate_ci_low", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("win_rate_ci_high", sa.Float(), nullable=False, server_default="0"),
    )
    op.add_column(
        "game_summaries",
        sa.Column("low_sample", sa.Boolean(), nullable=False, server_default="1"),
    )


def downgrade() -> None:
    op.drop_column("game_summaries", "low_sample")
    op.drop_column("game_summaries", "win_rate_ci_high")
    op.drop_column("game_summaries", "win_rate_ci_low")
    op.drop_column("game_summaries", "win_rate")
    op.drop_column("game_summaries", "malformed_rate_ci_high")
    op.drop_column("game_summaries", "malformed_rate_ci_low")
    op.drop_column("game_summaries", "illegal_rate_ci_high")
    op.drop_column("game_summaries", "illegal_rate_ci_low")
    op.drop_column("game_summaries", "malformed_attempts")
    op.drop_column("game_summaries", "illegal_attempts")
    op.drop_column("game_summaries", "attempt_count")
    op.drop_column("game_summaries", "accuracy_rate")
    op.drop_column("game_summaries", "evaluated_move_count")

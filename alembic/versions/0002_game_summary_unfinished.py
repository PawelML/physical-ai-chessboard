"""add unfinished game summary counter

Revision ID: 0002_game_summary_unfinished
Revises: 0001_initial
Create Date: 2026-06-06
"""
from collections.abc import Sequence

revision: str = "0002_game_summary_unfinished"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Current 0001_initial already includes this column. Keep this revision as a
    # no-op so fresh Alembic upgrades can replay the full history.
    pass


def downgrade() -> None:
    pass

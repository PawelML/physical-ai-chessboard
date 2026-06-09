"""add app_settings key-value table

Revision ID: 0004_app_settings
Revises: 0003_attempt_thinking
Create Date: 2026-06-09
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0004_app_settings"
down_revision: str | None = "0003_attempt_thinking"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "app_settings",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column("value", sa.JSON(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("app_settings")

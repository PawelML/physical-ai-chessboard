"""record per-attempt reranker metadata

Revision ID: 0007_attempt_reranker_metadata
Revises: 0006_summary_statistics
Create Date: 2026-06-15
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

from alembic import op

revision: str = "0007_attempt_reranker_metadata"
down_revision: str | None = "0006_summary_statistics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("attempts", sa.Column("reranker_metadata", sqlite.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("attempts", "reranker_metadata")

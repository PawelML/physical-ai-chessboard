"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-06
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("family", sa.String(length=255), nullable=True),
        sa.Column("param_size", sa.String(length=64), nullable=True),
        sa.Column("modality", sa.String(length=32), nullable=False),
        sa.Column("is_local", sa.Boolean(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("template_hash", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("legality_mode", sa.String(length=32), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "opening_suites",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "model_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("ollama_digest", sa.String(length=255), nullable=True),
        sa.Column("quantization", sa.String(length=64), nullable=True),
        sa.Column("context_window", sa.Integer(), nullable=True),
        sa.Column("sampler_params", sqlite.JSON(), nullable=True),
        sa.Column("runtime_version", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "opening_lines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("suite_id", sa.Integer(), nullable=False),
        sa.Column("eco", sa.String(length=16), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("start_fen", sa.Text(), nullable=True),
        sa.Column("move_sequence", sa.Text(), nullable=True),
        sa.Column("intended_ply_start", sa.Integer(), nullable=False),
        sa.Column("pairing_id", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["suite_id"], ["opening_suites.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("config_hash", sa.String(length=64), nullable=False),
        sa.Column("git_commit", sa.String(length=64), nullable=True),
        sa.Column("hardware_label", sa.String(length=255), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("stockfish_version", sa.String(length=255), nullable=True),
        sa.Column("stockfish_options", sqlite.JSON(), nullable=True),
        sa.Column("prompt_id", sa.Integer(), nullable=True),
        sa.Column("opening_suite_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["opening_suite_id"], ["opening_suites.id"]),
        sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "run_participants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("model_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("opponent_type", sa.String(length=32), nullable=False),
        sa.Column("stockfish_skill", sa.Integer(), nullable=True),
        sa.Column("uci_limit_strength", sa.Boolean(), nullable=True),
        sa.Column("target_elo", sa.Integer(), nullable=True),
        sa.Column("color_policy", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.ForeignKeyConstraint(["model_snapshot_id"], ["model_snapshots.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["benchmark_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "games",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("white_participant_id", sa.Integer(), nullable=True),
        sa.Column("black_participant_id", sa.Integer(), nullable=True),
        sa.Column("opening_line_id", sa.Integer(), nullable=True),
        sa.Column("result", sa.String(length=16), nullable=False),
        sa.Column("termination_reason", sa.String(length=64), nullable=True),
        sa.Column("final_fen", sa.Text(), nullable=True),
        sa.Column("pgn", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["black_participant_id"], ["run_participants.id"]),
        sa.ForeignKeyConstraint(["opening_line_id"], ["opening_lines.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["benchmark_runs.id"]),
        sa.ForeignKeyConstraint(["white_participant_id"], ["run_participants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "moves",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("ply", sa.Integer(), nullable=False),
        sa.Column("color", sa.String(length=8), nullable=False),
        sa.Column("fen_before", sa.Text(), nullable=False),
        sa.Column("fen_after", sa.Text(), nullable=False),
        sa.Column("accepted_uci", sa.String(length=16), nullable=False),
        sa.Column("accepted_san", sa.String(length=32), nullable=False),
        sa.Column("legal_move_count", sa.Integer(), nullable=False),
        sa.Column("move_source", sa.String(length=32), nullable=False),
        sa.Column("retries_used", sa.Integer(), nullable=False),
        sa.Column("latency_total_ms", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "attempts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("move_id", sa.Integer(), nullable=True),
        sa.Column("game_id", sa.Integer(), nullable=False),
        sa.Column("ply", sa.Integer(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Integer(), nullable=True),
        sa.Column("raw_prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("raw_prompt", sa.Text(), nullable=True),
        sa.Column("raw_response", sa.Text(), nullable=False),
        sa.Column("parsed_move", sa.String(length=16), nullable=True),
        sa.Column("parse_ok", sa.Boolean(), nullable=False),
        sa.Column("legal_ok", sa.Boolean(), nullable=False),
        sa.Column("error_type", sa.String(length=64), nullable=True),
        sa.Column("feedback_given", sqlite.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["game_id"], ["games.id"]),
        sa.ForeignKeyConstraint(["move_id"], ["moves.id"]),
        sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "token_usage",
        sa.Column("attempt_id", sa.Integer(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_context_window", sa.Integer(), nullable=False),
        sa.Column("estimated_context_remaining", sa.Integer(), nullable=False),
        sa.Column("truncation_applied", sa.Boolean(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["attempt_id"], ["attempts.id"]),
        sa.PrimaryKeyConstraint("attempt_id"),
    )
    op.create_table(
        "engine_evaluations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("move_id", sa.Integer(), nullable=False),
        sa.Column("engine_name", sa.String(length=64), nullable=False),
        sa.Column("engine_version", sa.String(length=255), nullable=False),
        sa.Column("nodes", sa.Integer(), nullable=False),
        sa.Column("depth_reached", sa.Integer(), nullable=True),
        sa.Column("eval_before_cp", sa.Integer(), nullable=True),
        sa.Column("eval_after_cp", sa.Integer(), nullable=True),
        sa.Column("mate_before", sa.Integer(), nullable=True),
        sa.Column("mate_after", sa.Integer(), nullable=True),
        sa.Column("best_move_uci", sa.String(length=16), nullable=True),
        sa.Column("centipawn_loss", sa.Integer(), nullable=True),
        sa.Column("classification", sa.String(length=32), nullable=False),
        sa.ForeignKeyConstraint(["move_id"], ["moves.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "move_annotations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("move_id", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Integer(), nullable=True),
        sa.Column("persona", sa.String(length=64), nullable=False),
        sa.Column("commentary", sa.Text(), nullable=False),
        sa.Column("raw_prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("raw_prompt", sa.Text(), nullable=True),
        sa.Column("raw_response", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["move_id"], ["moves.id"]),
        sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "game_summaries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("run_participant_id", sa.Integer(), nullable=False),
        sa.Column("model_snapshot_id", sa.Integer(), nullable=True),
        sa.Column("color", sa.String(length=8), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("legality_mode", sa.String(length=32), nullable=False),
        sa.Column("opening_suite_id", sa.Integer(), nullable=True),
        sa.Column("games_played", sa.Integer(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=False),
        sa.Column("draws", sa.Integer(), nullable=False),
        sa.Column("losses", sa.Integer(), nullable=False),
        sa.Column("unfinished", sa.Integer(), nullable=False),
        sa.Column("avg_cpl", sa.Float(), nullable=True),
        sa.Column("blunders", sa.Integer(), nullable=False),
        sa.Column("mistakes", sa.Integer(), nullable=False),
        sa.Column("inaccuracies", sa.Integer(), nullable=False),
        sa.Column("illegal_rate", sa.Float(), nullable=False),
        sa.Column("malformed_rate", sa.Float(), nullable=False),
        sa.Column("avg_retries", sa.Float(), nullable=False),
        sa.Column("forfeit_invalid_count", sa.Integer(), nullable=False),
        sa.Column("avg_latency_ms", sa.Float(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["model_snapshot_id"], ["model_snapshots.id"]),
        sa.ForeignKeyConstraint(["opening_suite_id"], ["opening_suites.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["benchmark_runs.id"]),
        sa.ForeignKeyConstraint(["run_participant_id"], ["run_participants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "operational_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("event_kind", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload", sqlite.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["benchmark_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("operational_events")
    op.drop_table("game_summaries")
    op.drop_table("move_annotations")
    op.drop_table("engine_evaluations")
    op.drop_table("token_usage")
    op.drop_table("attempts")
    op.drop_table("moves")
    op.drop_table("games")
    op.drop_table("run_participants")
    op.drop_table("benchmark_runs")
    op.drop_table("opening_lines")
    op.drop_table("model_snapshots")
    op.drop_table("opening_suites")
    op.drop_table("prompts")
    op.drop_table("models")

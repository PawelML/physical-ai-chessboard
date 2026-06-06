from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    family: Mapped[str | None] = mapped_column(String(255))
    param_size: Mapped[str | None] = mapped_column(String(64))
    modality: Mapped[str] = mapped_column(String(32), default="text", nullable=False)
    is_local: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


class ModelSnapshot(Base):
    __tablename__ = "model_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    ollama_digest: Mapped[str | None] = mapped_column(String(255))
    quantization: Mapped[str | None] = mapped_column(String(64))
    context_window: Mapped[int | None] = mapped_column(Integer)
    sampler_params: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    runtime_version: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    template_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    legality_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)


class OpeningSuite(Base):
    __tablename__ = "opening_suites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    lines: Mapped[list["OpeningLine"]] = relationship(back_populates="suite")


class OpeningLine(Base):
    __tablename__ = "opening_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    suite_id: Mapped[int] = mapped_column(ForeignKey("opening_suites.id"), nullable=False)
    eco: Mapped[str | None] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    start_fen: Mapped[str | None] = mapped_column(Text)
    move_sequence: Mapped[str | None] = mapped_column(Text)
    intended_ply_start: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pairing_id: Mapped[str | None] = mapped_column(String(64))

    suite: Mapped[OpeningSuite] = relationship(back_populates="lines")
    games: Mapped[list["Game"]] = relationship(back_populates="opening_line")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    hardware_label: Mapped[str | None] = mapped_column(String(255))
    seed: Mapped[int | None] = mapped_column(Integer)
    stockfish_version: Mapped[str | None] = mapped_column(String(255))
    stockfish_options: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompts.id"))
    opening_suite_id: Mapped[int | None] = mapped_column(ForeignKey("opening_suites.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    participants: Mapped[list["RunParticipant"]] = relationship(back_populates="run")
    games: Mapped[list["Game"]] = relationship(back_populates="run")
    operational_events: Mapped[list["OperationalEvent"]] = relationship(back_populates="run")


class RunParticipant(Base):
    __tablename__ = "run_participants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id"), nullable=False)
    model_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("model_snapshots.id"))
    opponent_type: Mapped[str] = mapped_column(String(32), nullable=False)
    stockfish_skill: Mapped[int | None] = mapped_column(Integer)
    uci_limit_strength: Mapped[bool | None] = mapped_column(Boolean)
    target_elo: Mapped[int | None] = mapped_column(Integer)
    color_policy: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)

    run: Mapped[BenchmarkRun] = relationship(back_populates="participants")
    white_games: Mapped[list["Game"]] = relationship(
        foreign_keys="Game.white_participant_id",
        back_populates="white_participant",
    )
    black_games: Mapped[list["Game"]] = relationship(
        foreign_keys="Game.black_participant_id",
        back_populates="black_participant",
    )


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("benchmark_runs.id"))
    white_participant_id: Mapped[int | None] = mapped_column(ForeignKey("run_participants.id"))
    black_participant_id: Mapped[int | None] = mapped_column(ForeignKey("run_participants.id"))
    opening_line_id: Mapped[int | None] = mapped_column(ForeignKey("opening_lines.id"))
    result: Mapped[str] = mapped_column(String(16), default="*", nullable=False)
    termination_reason: Mapped[str | None] = mapped_column(String(64))
    final_fen: Mapped[str | None] = mapped_column(Text)
    pgn: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    moves: Mapped[list["Move"]] = relationship(back_populates="game")
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="game")
    run: Mapped[BenchmarkRun | None] = relationship(back_populates="games")
    opening_line: Mapped[OpeningLine | None] = relationship(back_populates="games")
    white_participant: Mapped[RunParticipant | None] = relationship(
        foreign_keys=[white_participant_id],
        back_populates="white_games",
    )
    black_participant: Mapped[RunParticipant | None] = relationship(
        foreign_keys=[black_participant_id],
        back_populates="black_games",
    )


class Move(Base):
    __tablename__ = "moves"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    ply: Mapped[int] = mapped_column(Integer, nullable=False)
    color: Mapped[str] = mapped_column(String(8), nullable=False)
    fen_before: Mapped[str] = mapped_column(Text, nullable=False)
    fen_after: Mapped[str] = mapped_column(Text, nullable=False)
    accepted_uci: Mapped[str] = mapped_column(String(16), nullable=False)
    accepted_san: Mapped[str] = mapped_column(String(32), nullable=False)
    legal_move_count: Mapped[int] = mapped_column(Integer, nullable=False)
    move_source: Mapped[str] = mapped_column(String(32), nullable=False)
    retries_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_total_ms: Mapped[float] = mapped_column(Float, default=0, nullable=False)

    game: Mapped[Game] = relationship(back_populates="moves")
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="move")
    engine_evaluations: Mapped[list["EngineEvaluation"]] = relationship(back_populates="move")
    annotations: Mapped[list["MoveAnnotation"]] = relationship(back_populates="move")


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    move_id: Mapped[int | None] = mapped_column(ForeignKey("moves.id"))
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False)
    ply: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompts.id"))
    raw_prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_prompt: Mapped[str | None] = mapped_column(Text)
    raw_response: Mapped[str] = mapped_column(Text, nullable=False)
    parsed_move: Mapped[str | None] = mapped_column(String(16))
    parse_ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    legal_ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(64))
    feedback_given: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    game: Mapped[Game] = relationship(back_populates="attempts")
    move: Mapped[Move | None] = relationship(back_populates="attempts")
    token_usage: Mapped["TokenUsage"] = relationship(back_populates="attempt")


class TokenUsage(Base):
    __tablename__ = "token_usage"

    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"), primary_key=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_context_remaining: Mapped[int] = mapped_column(Integer, nullable=False)
    truncation_applied: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cost_usd: Mapped[float | None] = mapped_column(Float)

    attempt: Mapped[Attempt] = relationship(back_populates="token_usage")


class EngineEvaluation(Base):
    __tablename__ = "engine_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    move_id: Mapped[int] = mapped_column(ForeignKey("moves.id"), nullable=False)
    engine_name: Mapped[str] = mapped_column(String(64), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(255), nullable=False)
    nodes: Mapped[int] = mapped_column(Integer, nullable=False)
    depth_reached: Mapped[int | None] = mapped_column(Integer)
    eval_before_cp: Mapped[int | None] = mapped_column(Integer)
    eval_after_cp: Mapped[int | None] = mapped_column(Integer)
    mate_before: Mapped[int | None] = mapped_column(Integer)
    mate_after: Mapped[int | None] = mapped_column(Integer)
    best_move_uci: Mapped[str | None] = mapped_column(String(16))
    centipawn_loss: Mapped[int | None] = mapped_column(Integer)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)

    move: Mapped[Move] = relationship(back_populates="engine_evaluations")


class MoveAnnotation(Base):
    __tablename__ = "move_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    move_id: Mapped[int] = mapped_column(ForeignKey("moves.id"), nullable=False)
    prompt_id: Mapped[int | None] = mapped_column(ForeignKey("prompts.id"))
    persona: Mapped[str] = mapped_column(String(64), nullable=False)
    commentary: Mapped[str] = mapped_column(Text, nullable=False)
    raw_prompt_hash: Mapped[str | None] = mapped_column(String(64))
    raw_prompt: Mapped[str | None] = mapped_column(Text)
    raw_response: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    move: Mapped[Move] = relationship(back_populates="annotations")


class GameSummary(Base):
    __tablename__ = "game_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("benchmark_runs.id"), nullable=False)
    run_participant_id: Mapped[int] = mapped_column(
        ForeignKey("run_participants.id"),
        nullable=False,
    )
    model_snapshot_id: Mapped[int | None] = mapped_column(ForeignKey("model_snapshots.id"))
    color: Mapped[str] = mapped_column(String(8), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    legality_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    opening_suite_id: Mapped[int | None] = mapped_column(ForeignKey("opening_suites.id"))
    games_played: Mapped[int] = mapped_column(Integer, nullable=False)
    wins: Mapped[int] = mapped_column(Integer, nullable=False)
    draws: Mapped[int] = mapped_column(Integer, nullable=False)
    losses: Mapped[int] = mapped_column(Integer, nullable=False)
    unfinished: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_cpl: Mapped[float | None] = mapped_column(Float)
    blunders: Mapped[int] = mapped_column(Integer, nullable=False)
    mistakes: Mapped[int] = mapped_column(Integer, nullable=False)
    inaccuracies: Mapped[int] = mapped_column(Integer, nullable=False)
    illegal_rate: Mapped[float] = mapped_column(Float, nullable=False)
    malformed_rate: Mapped[float] = mapped_column(Float, nullable=False)
    avg_retries: Mapped[float] = mapped_column(Float, nullable=False)
    forfeit_invalid_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)


class OperationalEvent(Base):
    __tablename__ = "operational_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("benchmark_runs.id"))
    event_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[BenchmarkRun | None] = relationship(back_populates="operational_events")

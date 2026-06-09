import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import chess
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arena_core.cli import _play_async, _source_from_name
from arena_core.config import Settings, get_settings
from arena_core.engine import ArenaGame, MoveSource
from arena_core.evaluators.stockfish import StockfishEvaluator
from arena_core.leaderboards import rebuild_game_summaries
from arena_core.move_sources import MoveProposal
from arena_core.persistence.database import create_session_factory
from arena_core.persistence.database import init_db as create_tables
from arena_core.persistence.models import (
    AppSetting,
    Attempt,
    BenchmarkRun,
    EngineEvaluation,
    Game,
    GameSummary,
    Move,
    MoveAnnotation,
    OperationalEvent,
    RunParticipant,
    TokenUsage,
)
from arena_core.reports import export_game_report
from arena_core.tournaments import TournamentConfig, run_tournament

GameStreamPayload = dict[str, list[dict[str, int | str | None]]]
GameJobStatus = Literal["running", "completed", "failed", "cancelled"]
GuidanceMode = Literal["legal_list", "strategic_memory"]
JobKind = Literal["game", "stockfish_match"]
StockfishLevel = Literal["beginner", "club"]
HumanColor = Literal["white", "black"]

GAME_DEFAULTS_KEY = "game_defaults"


class GameDefaults(BaseModel):
    """Per-model sampling/runtime knobs surfaced in the UI and saved as defaults."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    num_ctx: int | None = Field(default=None, gt=0)
    num_predict: int | None = Field(default=None, gt=0)


class GameJob(BaseModel):
    id: str
    status: GameJobStatus
    kind: JobKind = "game"
    white: str
    black: str
    legality_mode: str
    temperature: float = 0.0
    top_p: float | None = None
    num_ctx: int | None = None
    num_predict: int | None = None
    ollama_thinking: bool = False
    ollama_cpu_offload: bool = False
    guidance_mode: GuidanceMode
    max_plies: int | None
    stockfish_level: StockfishLevel | None = None
    games_requested: int | None = None
    games_completed: int = 0
    run_id: int | None = None
    game_ids: list[int] = Field(default_factory=list)
    game_id: int | None = None
    result: str | None = None
    termination_reason: str | None = None
    error: str | None = None
    created_at: str
    completed_at: str | None = None


class ModelOption(BaseModel):
    id: str
    label: str
    provider: str


class StartGameRequest(BaseModel):
    white: str
    black: str
    legality_mode: Literal["open", "constrained"] = "constrained"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    num_ctx: int | None = Field(default=None, gt=0)
    num_predict: int | None = Field(default=None, gt=0)
    ollama_thinking: bool = False
    ollama_cpu_offload: bool = False
    guidance_mode: GuidanceMode = "legal_list"
    max_plies: int | None = None
    stockfish_path: str | None = None


class StartStockfishMatchRequest(BaseModel):
    model: str
    stockfish_level: StockfishLevel = "beginner"
    game_count: int = Field(default=4, ge=1, le=200)
    legality_mode: Literal["open", "constrained"] = "constrained"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    num_ctx: int | None = Field(default=None, gt=0)
    num_predict: int | None = Field(default=None, gt=0)
    ollama_thinking: bool = False
    ollama_cpu_offload: bool = False
    guidance_mode: GuidanceMode = "legal_list"
    max_plies: int | None = None
    stockfish_path: str | None = None


class StartHumanGameRequest(BaseModel):
    human_color: HumanColor = "white"
    opponent: str
    stockfish_level: StockfishLevel | None = None
    legality_mode: Literal["open", "constrained"] = "constrained"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    num_ctx: int | None = Field(default=None, gt=0)
    num_predict: int | None = Field(default=None, gt=0)
    ollama_thinking: bool = False
    ollama_cpu_offload: bool = False
    guidance_mode: GuidanceMode = "legal_list"
    max_plies: int | None = None
    stockfish_path: str | None = None


class HumanMoveRequest(BaseModel):
    move: str


class StartGameResponse(BaseModel):
    job_id: str
    status: GameJobStatus


class CancelGameResponse(BaseModel):
    job_id: str
    status: GameJobStatus
    game_id: int | None = None


class HumanGameState(BaseModel):
    id: str
    status: GameJobStatus
    game_id: int
    human_color: HumanColor
    opponent: str
    turn: HumanColor | None
    result: str | None = None
    termination_reason: str | None = None
    legal_moves: list[str]
    error: str | None = None
    created_at: str
    completed_at: str | None = None


@dataclass
class HumanGameRuntime:
    arena: ArenaGame
    game_id: int
    human_color: chess.Color
    opponent_source: MoveSource
    lock: asyncio.Lock


class GpuTelemetry(BaseModel):
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_percent: int | None


class OllamaRuntimeModel(BaseModel):
    name: str
    size_bytes: int | None = None
    size_vram_bytes: int | None = None
    size_cpu_bytes: int | None = None
    vram_percent: float | None = None
    offload_status: str
    processor: str | None = None
    context_window: int | None = None
    expires_at: str | None = None


class RuntimeTelemetry(BaseModel):
    sampled_at: str
    gpus: list[GpuTelemetry]
    ollama_models: list[OllamaRuntimeModel]


class GameListItem(BaseModel):
    id: int
    run_id: int | None
    result: str
    termination_reason: str | None
    final_fen: str | None
    started_at: str
    ended_at: str | None


class TokenUsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_context_window: int
    estimated_context_remaining: int
    truncation_applied: bool
    cost_usd: float | None


class AttemptOut(BaseModel):
    id: int
    ply: int
    attempt_number: int
    parsed_move: str | None
    parse_ok: bool
    legal_ok: bool
    error_type: str | None
    latency_ms: float
    token_usage: TokenUsageOut | None


class EngineEvaluationOut(BaseModel):
    engine_name: str
    engine_version: str
    nodes: int
    depth_reached: int | None
    eval_before_cp: int | None
    eval_after_cp: int | None
    mate_before: int | None
    mate_after: int | None
    best_move_uci: str | None
    centipawn_loss: int | None
    classification: str


class MoveAnnotationOut(BaseModel):
    persona: str
    commentary: str
    created_at: str


class MoveOut(BaseModel):
    id: int
    ply: int
    color: str
    fen_before: str
    fen_after: str
    accepted_uci: str
    accepted_san: str
    legal_move_count: int
    move_source: str
    retries_used: int
    latency_total_ms: float
    attempts: list[AttemptOut]
    engine_evaluations: list[EngineEvaluationOut]
    annotations: list[MoveAnnotationOut]


class GameDetail(BaseModel):
    id: int
    run_id: int | None
    white_participant_id: int | None
    black_participant_id: int | None
    white_player: str | None
    black_player: str | None
    result: str
    termination_reason: str | None
    final_fen: str | None
    pgn: str | None
    moves: list[MoveOut]


class RunListItem(BaseModel):
    id: int
    name: str
    config_hash: str
    seed: int | None
    opening_suite_id: int | None
    created_at: str


class LeaderboardRow(BaseModel):
    id: int
    run_id: int
    run_participant_id: int
    participant: str
    model_snapshot_id: int | None
    color: str
    mode: str
    legality_mode: str
    opening_suite_id: int | None
    games_played: int
    wins: int
    draws: int
    losses: int
    unfinished: int
    avg_game_plies: float
    avg_cpl: float | None
    blunders: int
    mistakes: int
    inaccuracies: int
    illegal_rate: float
    malformed_rate: float
    avg_retries: float
    forfeit_invalid_count: int
    avg_latency_ms: float
    total_tokens: int


class OperationalEventOut(BaseModel):
    id: int
    run_id: int | None
    event_kind: str
    severity: str
    message: str
    payload: dict[str, object] | None
    created_at: str


class RunComparisonRow(BaseModel):
    run_id: int
    games_played: int
    wins: int
    draws: int
    losses: int
    unfinished: int
    avg_game_plies: float
    avg_cpl: float | None
    illegal_rate: float
    malformed_rate: float
    avg_retries: float
    avg_latency_ms: float
    total_tokens: int


def create_app(settings: Settings | None = None) -> FastAPI:
    effective_settings = settings or get_settings()
    session_factory = create_session_factory(effective_settings.database_url)
    api = FastAPI(title="Physical AI Chessboard Arena")
    game_jobs: dict[str, GameJob] = {}
    game_tasks: dict[str, asyncio.Task[None]] = {}
    human_games: dict[str, HumanGameState] = {}
    human_runtimes: dict[str, HumanGameRuntime] = {}

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/models")
    async def list_models() -> list[ModelOption]:
        options = [ModelOption(id="random", label="random", provider="built-in")]
        if _effective_stockfish_path(effective_settings, None):
            options.append(ModelOption(id="stockfish", label="stockfish", provider="engine"))
        options.extend(_gemini_model_options(effective_settings))
        options.extend(await _ollama_model_options(effective_settings))
        return options

    @api.get("/runtime/telemetry")
    async def runtime_telemetry() -> RuntimeTelemetry:
        gpus, ollama_models = await asyncio.gather(
            _gpu_telemetry(),
            _ollama_runtime_models(effective_settings),
        )
        return RuntimeTelemetry(
            sampled_at=_utcnow_iso(),
            gpus=gpus,
            ollama_models=ollama_models,
        )

    @api.get("/games/jobs")
    async def list_game_jobs() -> list[GameJob]:
        return sorted(game_jobs.values(), key=lambda job: job.created_at, reverse=True)

    @api.get("/human-games")
    async def list_human_games() -> list[HumanGameState]:
        return sorted(human_games.values(), key=lambda game: game.created_at, reverse=True)

    @api.get("/settings/game-defaults")
    async def get_game_defaults() -> GameDefaults:
        async with session_factory() as session:
            row = await session.get(AppSetting, GAME_DEFAULTS_KEY)
            if row is None:
                return GameDefaults()
            return GameDefaults.model_validate(row.value)

    @api.put("/settings/game-defaults")
    async def put_game_defaults(payload: GameDefaults) -> GameDefaults:
        async with session_factory() as session:
            async with session.begin():
                row = await session.get(AppSetting, GAME_DEFAULTS_KEY)
                if row is None:
                    session.add(AppSetting(key=GAME_DEFAULTS_KEY, value=payload.model_dump()))
                else:
                    row.value = payload.model_dump()
        return payload

    @api.post("/games/start")
    async def start_game(payload: StartGameRequest) -> StartGameResponse:
        white = payload.white.strip()
        black = payload.black.strip()
        if not white or not black:
            raise HTTPException(status_code=400, detail="white and black sources are required")
        if payload.max_plies is not None and payload.max_plies <= 0:
            raise HTTPException(status_code=400, detail="max_plies must be greater than zero")

        job_id = uuid.uuid4().hex
        job = GameJob(
            id=job_id,
            status="running",
            white=white,
            black=black,
            legality_mode=payload.legality_mode,
            temperature=payload.temperature,
            top_p=payload.top_p,
            num_ctx=payload.num_ctx,
            num_predict=payload.num_predict,
            ollama_thinking=payload.ollama_thinking,
            ollama_cpu_offload=payload.ollama_cpu_offload,
            guidance_mode=payload.guidance_mode,
            max_plies=payload.max_plies,
            created_at=_utcnow_iso(),
        )
        game_jobs[job_id] = job
        game_tasks[job_id] = asyncio.create_task(
            _run_game_job(
                job_id=job_id,
                jobs=game_jobs,
                tasks=game_tasks,
                session_factory=session_factory,
                settings=effective_settings,
                white=white,
                black=black,
                legality_mode=payload.legality_mode,
                sampling=GameDefaults(
                    temperature=payload.temperature,
                    top_p=payload.top_p,
                    num_ctx=payload.num_ctx,
                    num_predict=payload.num_predict,
                ),
                ollama_thinking=payload.ollama_thinking,
                ollama_cpu_offload=payload.ollama_cpu_offload,
                guidance_mode=payload.guidance_mode,
                max_plies=payload.max_plies,
                stockfish_path=payload.stockfish_path,
            )
        )
        return StartGameResponse(job_id=job_id, status="running")

    @api.post("/matches/stockfish/start")
    async def start_stockfish_match(payload: StartStockfishMatchRequest) -> StartGameResponse:
        model = payload.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="model source is required")
        if payload.max_plies is not None and payload.max_plies <= 0:
            raise HTTPException(status_code=400, detail="max_plies must be greater than zero")
        stockfish_path = _effective_stockfish_path(effective_settings, payload.stockfish_path)
        if stockfish_path is None:
            raise HTTPException(
                status_code=400,
                detail="stockfish binary not configured; set ARENA_STOCKFISH_PATH",
            )

        job_id = uuid.uuid4().hex
        stockfish_label = _stockfish_display_name(payload.stockfish_level)
        job = GameJob(
            id=job_id,
            status="running",
            kind="stockfish_match",
            white=model,
            black=stockfish_label,
            legality_mode=payload.legality_mode,
            temperature=payload.temperature,
            top_p=payload.top_p,
            num_ctx=payload.num_ctx,
            num_predict=payload.num_predict,
            ollama_thinking=payload.ollama_thinking,
            ollama_cpu_offload=payload.ollama_cpu_offload,
            guidance_mode=payload.guidance_mode,
            max_plies=payload.max_plies,
            stockfish_level=payload.stockfish_level,
            games_requested=payload.game_count,
            created_at=_utcnow_iso(),
        )
        game_jobs[job_id] = job
        game_tasks[job_id] = asyncio.create_task(
            _run_stockfish_match_job(
                job_id=job_id,
                jobs=game_jobs,
                tasks=game_tasks,
                session_factory=session_factory,
                settings=effective_settings,
                model=model,
                level=payload.stockfish_level,
                game_count=payload.game_count,
                legality_mode=payload.legality_mode,
                sampling=GameDefaults(
                    temperature=payload.temperature,
                    top_p=payload.top_p,
                    num_ctx=payload.num_ctx,
                    num_predict=payload.num_predict,
                ),
                ollama_thinking=payload.ollama_thinking,
                ollama_cpu_offload=payload.ollama_cpu_offload,
                guidance_mode=payload.guidance_mode,
                max_plies=payload.max_plies,
                stockfish_path=stockfish_path,
            )
        )
        return StartGameResponse(job_id=job_id, status="running")

    @api.post("/human-games/start")
    async def start_human_game(payload: StartHumanGameRequest) -> HumanGameState:
        opponent = payload.opponent.strip()
        if not opponent:
            raise HTTPException(status_code=400, detail="opponent source is required")
        if payload.max_plies is not None and payload.max_plies <= 0:
            raise HTTPException(status_code=400, detail="max_plies must be greater than zero")
        await create_tables(effective_settings.database_url)

        base_settings = _settings_for_ollama_options(
            effective_settings,
            sampling=GameDefaults(
                temperature=payload.temperature,
                top_p=payload.top_p,
                num_ctx=payload.num_ctx,
                num_predict=payload.num_predict,
            ),
            thinking=payload.ollama_thinking,
            cpu_offload=payload.ollama_cpu_offload,
        )
        stockfish_path = _effective_stockfish_path(effective_settings, payload.stockfish_path)
        game_settings = base_settings
        display_opponent = opponent
        if opponent == "stockfish":
            if stockfish_path is None:
                raise HTTPException(
                    status_code=400,
                    detail="stockfish binary not configured; set ARENA_STOCKFISH_PATH",
                )
            game_settings = _settings_for_stockfish_level(
                base_settings,
                level=payload.stockfish_level or "beginner",
                stockfish_path=stockfish_path,
            )
            display_opponent = _stockfish_display_name(payload.stockfish_level or "beginner")
        elif stockfish_path is not None:
            game_settings = base_settings.model_copy(update={"stockfish_path": stockfish_path})

        opponent_source = _source_from_name(opponent, game_settings)
        white: MoveSource
        black: MoveSource
        if payload.human_color == "white":
            white = _HumanApiMoveSource("human")
            black = opponent_source
        else:
            white = opponent_source
            black = _HumanApiMoveSource("human")
        evaluator = (
            StockfishEvaluator(
                binary_path=stockfish_path,
                nodes=game_settings.stockfish_nodes,
                threads=game_settings.stockfish_threads,
                hash_mb=game_settings.stockfish_hash_mb,
            )
            if stockfish_path is not None
            else None
        )
        arena = ArenaGame(
            white=white,
            black=black,
            settings=game_settings,
            legality_mode=payload.legality_mode,
            max_plies=payload.max_plies,
            evaluator=evaluator,
            strategic_memory=payload.guidance_mode == "strategic_memory",
        )
        human_game_id = uuid.uuid4().hex
        async with session_factory() as session:
            game_row = await arena.start(session)
            await session.commit()
            state = _human_game_state(
                human_game_id=human_game_id,
                arena=arena,
                game_id=game_row.id,
                human_color=_color_from_name(payload.human_color),
                opponent=display_opponent,
                status="running",
            )
        human_games[human_game_id] = state
        human_runtimes[human_game_id] = HumanGameRuntime(
            arena=arena,
            game_id=state.game_id,
            human_color=_color_from_name(payload.human_color),
            opponent_source=opponent_source,
            lock=asyncio.Lock(),
        )
        if payload.human_color == "black":
            state = await _play_opponent_until_human_turn(
                human_game_id,
                states=human_games,
                runtimes=human_runtimes,
                session_factory=session_factory,
            )
        return state

    @api.get("/human-games/{human_game_id}")
    async def get_human_game(human_game_id: str) -> HumanGameState:
        state = human_games.get(human_game_id)
        if state is None:
            raise HTTPException(status_code=404, detail="human game not found")
        return state

    @api.post("/human-games/{human_game_id}/move")
    async def play_human_move(human_game_id: str, payload: HumanMoveRequest) -> HumanGameState:
        runtime = human_runtimes.get(human_game_id)
        state = human_games.get(human_game_id)
        if runtime is None or state is None:
            raise HTTPException(status_code=404, detail="human game not found")
        if state.status != "running":
            return state
        async with runtime.lock:
            if runtime.arena.board.turn != runtime.human_color:
                raise HTTPException(status_code=409, detail="not the human player's turn")
            async with session_factory() as session:
                game_row = await session.get(Game, runtime.game_id)
                if game_row is None:
                    raise HTTPException(status_code=404, detail="game not found")
                ok, reason = await runtime.arena.play_human_move(
                    session,
                    runtime.game_id,
                    payload.move,
                )
                if not ok:
                    await session.commit()
                    human_games[human_game_id] = _human_game_state(
                        human_game_id=human_game_id,
                        arena=runtime.arena,
                        game_id=runtime.game_id,
                        human_color=runtime.human_color,
                        opponent=state.opponent,
                        status="running",
                        error=reason,
                        created_at=state.created_at,
                    )
                    return human_games[human_game_id]
                await _commit_or_finish_human_game(
                    session=session,
                    state_id=human_game_id,
                    states=human_games,
                    runtimes=human_runtimes,
                    runtime=runtime,
                    game_row=game_row,
                    opponent=state.opponent,
                )
        return await _play_opponent_until_human_turn(
            human_game_id,
            states=human_games,
            runtimes=human_runtimes,
            session_factory=session_factory,
        )

    @api.post("/human-games/{human_game_id}/cancel")
    async def cancel_human_game(human_game_id: str) -> HumanGameState:
        runtime = human_runtimes.get(human_game_id)
        state = human_games.get(human_game_id)
        if runtime is None or state is None:
            raise HTTPException(status_code=404, detail="human game not found")
        async with runtime.lock:
            async with session_factory() as session:
                game_row = await session.get(Game, runtime.game_id)
                if game_row is not None and game_row.ended_at is None:
                    runtime.arena.finish(
                        game_row,
                        termination_reason="aborted_by_user",
                        result="*",
                    )
                    await session.commit()
            _close_if_present(runtime.opponent_source)
            human_runtimes.pop(human_game_id, None)
            human_games[human_game_id] = state.model_copy(
                update={
                    "status": "cancelled",
                    "result": "*",
                    "termination_reason": "aborted_by_user",
                    "turn": None,
                    "legal_moves": [],
                    "completed_at": _utcnow_iso(),
                }
            )
            return human_games[human_game_id]

    @api.post("/games/jobs/{job_id}/cancel")
    async def cancel_game_job(job_id: str) -> CancelGameResponse:
        job = game_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="game job not found")
        if job.status != "running":
            return CancelGameResponse(job_id=job.id, status=job.status, game_id=job.game_id)

        if job.game_id is not None:
            await _mark_game_cancelled(session_factory, game_id=job.game_id)
        game_jobs[job_id] = job.model_copy(
            update={
                "status": "cancelled",
                "termination_reason": "aborted_by_user",
                "completed_at": _utcnow_iso(),
            }
        )
        task = game_tasks.get(job_id)
        if task is not None:
            task.cancel()
        return CancelGameResponse(job_id=job_id, status="cancelled", game_id=job.game_id)

    @api.get("/games")
    async def list_games() -> list[GameListItem]:
        async with session_factory() as session:
            rows = (
                await session.execute(select(Game).order_by(Game.started_at.desc(), Game.id.desc()))
            ).scalars()
            return [
                GameListItem(
                    id=row.id,
                    run_id=row.run_id,
                    result=row.result,
                    termination_reason=row.termination_reason,
                    final_fen=row.final_fen,
                    started_at=row.started_at.isoformat(),
                    ended_at=row.ended_at.isoformat() if row.ended_at else None,
                )
                for row in rows
            ]

    @api.get("/runs")
    async def list_runs() -> list[RunListItem]:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(BenchmarkRun).order_by(
                        BenchmarkRun.created_at.desc(),
                        BenchmarkRun.id.desc(),
                    )
                )
            ).scalars()
            return [
                RunListItem(
                    id=row.id,
                    name=row.name,
                    config_hash=row.config_hash,
                    seed=row.seed,
                    opening_suite_id=row.opening_suite_id,
                    created_at=row.created_at.isoformat(),
                )
                for row in rows
            ]

    @api.get("/runs/{run_id}/games")
    async def list_run_games(run_id: int) -> list[GameListItem]:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(Game).where(Game.run_id == run_id).order_by(Game.id)
                )
            ).scalars()
            return [
                GameListItem(
                    id=row.id,
                    run_id=row.run_id,
                    result=row.result,
                    termination_reason=row.termination_reason,
                    final_fen=row.final_fen,
                    started_at=row.started_at.isoformat(),
                    ended_at=row.ended_at.isoformat() if row.ended_at else None,
                )
                for row in rows
            ]

    @api.get("/runs/{run_id}/events")
    async def list_run_events(run_id: int) -> list[OperationalEventOut]:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    select(OperationalEvent)
                    .where(OperationalEvent.run_id == run_id)
                    .order_by(OperationalEvent.id)
                )
            ).scalars()
            return [_operational_event_out(row) for row in rows]

    @api.get("/runs/compare")
    async def compare_runs() -> list[RunComparisonRow]:
        summaries = await _all_summaries(session_factory)
        grouped: dict[int, list[GameSummary]] = {}
        for summary in summaries:
            grouped.setdefault(summary.run_id, []).append(summary)
        return [
            _comparison_row(run_id, rows)
            for run_id, rows in sorted(grouped.items(), reverse=True)
        ]

    @api.get("/leaderboard")
    async def leaderboard(
        run_id: int | None = None,
        color: str | None = None,
        mode: str | None = None,
        legality_mode: str | None = None,
        opening_suite_id: int | None = None,
        model_snapshot_id: int | None = None,
    ) -> list[LeaderboardRow]:
        async with session_factory() as session:
            query = select(GameSummary, RunParticipant).join(
                RunParticipant,
                RunParticipant.id == GameSummary.run_participant_id,
            )
            if run_id is not None:
                query = query.where(GameSummary.run_id == run_id)
            if color is not None:
                query = query.where(GameSummary.color == color)
            if mode is not None:
                query = query.where(GameSummary.mode == mode)
            if legality_mode is not None:
                query = query.where(GameSummary.legality_mode == legality_mode)
            if opening_suite_id is not None:
                query = query.where(GameSummary.opening_suite_id == opening_suite_id)
            if model_snapshot_id is not None:
                query = query.where(GameSummary.model_snapshot_id == model_snapshot_id)
            rows = (
                await session.execute(
                    query.order_by(
                        GameSummary.run_id.desc(),
                        GameSummary.legality_mode,
                        GameSummary.color,
                        GameSummary.avg_cpl.is_(None),
                        GameSummary.avg_cpl,
                        GameSummary.illegal_rate,
                    )
                )
            ).all()
            return [
                LeaderboardRow(
                    id=summary.id,
                    run_id=summary.run_id,
                    run_participant_id=summary.run_participant_id,
                    participant=participant.display_name,
                    model_snapshot_id=summary.model_snapshot_id,
                    color=summary.color,
                    mode=summary.mode,
                    legality_mode=summary.legality_mode,
                    opening_suite_id=summary.opening_suite_id,
                    games_played=summary.games_played,
                    wins=summary.wins,
                    draws=summary.draws,
                    losses=summary.losses,
                    unfinished=summary.unfinished,
                    avg_game_plies=summary.avg_game_plies,
                    avg_cpl=summary.avg_cpl,
                    blunders=summary.blunders,
                    mistakes=summary.mistakes,
                    inaccuracies=summary.inaccuracies,
                    illegal_rate=summary.illegal_rate,
                    malformed_rate=summary.malformed_rate,
                    avg_retries=summary.avg_retries,
                    forfeit_invalid_count=summary.forfeit_invalid_count,
                    avg_latency_ms=summary.avg_latency_ms,
                    total_tokens=summary.total_tokens,
                )
                for summary, participant in rows
            ]

    @api.get("/stream/games")
    async def stream_games(interval_seconds: float = 2.0) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            while True:
                payload = await _game_stream_payload(session_factory)
                yield f"event: games\ndata: {json.dumps(payload)}\n\n"
                await asyncio.sleep(max(interval_seconds, 0.25))

        return StreamingResponse(events(), media_type="text/event-stream")

    @api.get("/games/{game_id}")
    async def get_game(game_id: int) -> GameDetail:
        async with session_factory() as session:
            game = await session.get(Game, game_id)
            if game is None:
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="game not found")
            moves = (
                await session.execute(
                    select(Move).where(Move.game_id == game_id).order_by(Move.ply)
                )
            ).scalars()
            move_rows = list(moves)
            attempts_by_move = await _attempts_by_move(session_factory, game_id)
            evals_by_move = await _evaluations_by_move(
                session_factory,
                [move.id for move in move_rows],
            )
            annotations_by_move = await _annotations_by_move(
                session_factory,
                [move.id for move in move_rows],
            )
            white_participant = (
                await session.get(RunParticipant, game.white_participant_id)
                if game.white_participant_id is not None
                else None
            )
            black_participant = (
                await session.get(RunParticipant, game.black_participant_id)
                if game.black_participant_id is not None
                else None
            )
            return GameDetail(
                id=game.id,
                run_id=game.run_id,
                white_participant_id=game.white_participant_id,
                black_participant_id=game.black_participant_id,
                white_player=(
                    white_participant.display_name
                    if white_participant
                    else _pgn_header(game.pgn, "White")
                ),
                black_player=(
                    black_participant.display_name
                    if black_participant
                    else _pgn_header(game.pgn, "Black")
                ),
                result=game.result,
                termination_reason=game.termination_reason,
                final_fen=game.final_fen,
                pgn=game.pgn,
                moves=[
                    MoveOut(
                        id=move.id,
                        ply=move.ply,
                        color=move.color,
                        fen_before=move.fen_before,
                        fen_after=move.fen_after,
                        accepted_uci=move.accepted_uci,
                        accepted_san=move.accepted_san,
                        legal_move_count=move.legal_move_count,
                        move_source=move.move_source,
                        retries_used=move.retries_used,
                        latency_total_ms=move.latency_total_ms,
                        attempts=attempts_by_move.get(move.id, []),
                        engine_evaluations=evals_by_move.get(move.id, []),
                        annotations=annotations_by_move.get(move.id, []),
                    )
                    for move in move_rows
                ],
            )

    @api.get("/games/{game_id}/report", response_class=PlainTextResponse)
    async def get_game_report(game_id: int) -> str:
        async with session_factory() as session:
            return await export_game_report(session, game_id=game_id)

    return api


async def _game_stream_payload(
    session_factory: async_sessionmaker[AsyncSession],
) -> GameStreamPayload:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Game).order_by(Game.started_at.desc(), Game.id.desc()).limit(20)
            )
        ).scalars()
        games = [
            {
                "id": row.id,
                "run_id": row.run_id,
                "result": row.result,
                "termination_reason": row.termination_reason,
                "final_fen": row.final_fen,
                "started_at": row.started_at.isoformat(),
                "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            }
            for row in rows
        ]
    return {"games": games}


class _HumanApiMoveSource:
    source_type = "human"

    def __init__(self, name: str) -> None:
        self.name = name

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:  # pragma: no cover
        raise RuntimeError("human moves are submitted through /human-games/{id}/move")


def _color_from_name(color: HumanColor) -> chess.Color:
    return chess.WHITE if color == "white" else chess.BLACK


def _color_name(color: chess.Color) -> HumanColor:
    return "white" if color == chess.WHITE else "black"


def _human_game_state(
    *,
    human_game_id: str,
    arena: ArenaGame,
    game_id: int,
    human_color: chess.Color,
    opponent: str,
    status: GameJobStatus,
    result: str | None = None,
    termination_reason: str | None = None,
    error: str | None = None,
    created_at: str | None = None,
    completed_at: str | None = None,
) -> HumanGameState:
    turn = None if status != "running" else _color_name(arena.board.turn)
    return HumanGameState(
        id=human_game_id,
        status=status,
        game_id=game_id,
        human_color=_color_name(human_color),
        opponent=opponent,
        turn=turn,
        result=result,
        termination_reason=termination_reason,
        legal_moves=(
            sorted(move.uci() for move in arena.board.legal_moves) if status == "running" else []
        ),
        error=error,
        created_at=created_at or _utcnow_iso(),
        completed_at=completed_at,
    )


async def _commit_or_finish_human_game(
    *,
    session: AsyncSession,
    state_id: str,
    states: dict[str, HumanGameState],
    runtimes: dict[str, HumanGameRuntime],
    runtime: HumanGameRuntime,
    game_row: Game,
    opponent: str,
) -> HumanGameState:
    game_row.final_fen = runtime.arena.board.fen()
    game_row.pgn = runtime.arena._export_pgn("*")
    is_finished, termination_reason = runtime.arena.pending_result()
    if is_finished and termination_reason is not None:
        runtime.arena.finish(game_row, termination_reason=termination_reason)
        await session.commit()
        _close_if_present(runtime.opponent_source)
        runtimes.pop(state_id, None)
        states[state_id] = _human_game_state(
            human_game_id=state_id,
            arena=runtime.arena,
            game_id=runtime.game_id,
            human_color=runtime.human_color,
            opponent=opponent,
            status="completed",
            result=game_row.result,
            termination_reason=termination_reason,
            created_at=states[state_id].created_at,
            completed_at=_utcnow_iso(),
        )
        return states[state_id]

    await session.commit()
    states[state_id] = _human_game_state(
        human_game_id=state_id,
        arena=runtime.arena,
        game_id=runtime.game_id,
        human_color=runtime.human_color,
        opponent=opponent,
        status="running",
        created_at=states[state_id].created_at,
    )
    return states[state_id]


async def _play_opponent_until_human_turn(
    human_game_id: str,
    *,
    states: dict[str, HumanGameState],
    runtimes: dict[str, HumanGameRuntime],
    session_factory: async_sessionmaker[AsyncSession],
) -> HumanGameState:
    runtime = runtimes.get(human_game_id)
    state = states.get(human_game_id)
    if runtime is None or state is None:
        raise HTTPException(status_code=404, detail="human game not found")
    async with runtime.lock:
        while runtime.arena.board.turn != runtime.human_color:
            async with session_factory() as session:
                game_row = await session.get(Game, runtime.game_id)
                if game_row is None:
                    raise HTTPException(status_code=404, detail="game not found")
                accepted = await runtime.arena.play_source_move(
                    session,
                    runtime.game_id,
                    runtime.opponent_source,
                )
                if not accepted:
                    runtime.arena.finish(game_row, termination_reason="forfeit_invalid")
                    await session.commit()
                    _close_if_present(runtime.opponent_source)
                    runtimes.pop(human_game_id, None)
                    states[human_game_id] = _human_game_state(
                        human_game_id=human_game_id,
                        arena=runtime.arena,
                        game_id=runtime.game_id,
                        human_color=runtime.human_color,
                        opponent=state.opponent,
                        status="completed",
                        result=game_row.result,
                        termination_reason="forfeit_invalid",
                        created_at=state.created_at,
                        completed_at=_utcnow_iso(),
                    )
                    return states[human_game_id]
                new_state = await _commit_or_finish_human_game(
                    session=session,
                    state_id=human_game_id,
                    states=states,
                    runtimes=runtimes,
                    runtime=runtime,
                    game_row=game_row,
                    opponent=state.opponent,
                )
                if new_state.status != "running":
                    return new_state
    return states[human_game_id]


def _close_if_present(source: object) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()


async def _ollama_model_options(settings: Settings) -> list[ModelOption]:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
    except httpx.HTTPError:
        return []

    payload = response.json()
    options: list[ModelOption] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name:
            options.append(ModelOption(id=name, label=name, provider="ollama"))
    return sorted(options, key=lambda option: option.label)


def _gemini_model_options(settings: Settings) -> list[ModelOption]:
    if (
        not settings.api_providers_enabled
        or not settings.gemini_api_key
        or not settings.gemini_model
    ):
        return []
    return [
        ModelOption(
            id=f"gemini:{settings.gemini_model}",
            label=settings.gemini_model,
            provider="gemini",
        )
    ]


async def _gpu_telemetry() -> list[GpuTelemetry]:
    if shutil.which("nvidia-smi") is None:
        return []
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _stderr = await process.communicate()
    if process.returncode != 0:
        return []

    gpus: list[GpuTelemetry] = []
    for line in stdout.decode().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuTelemetry(
                    name=parts[0],
                    memory_used_mb=int(parts[1]),
                    memory_total_mb=int(parts[2]),
                    utilization_percent=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return gpus


async def _ollama_runtime_models(settings: Settings) -> list[OllamaRuntimeModel]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/ps")
            response.raise_for_status()
    except httpx.HTTPError:
        return []

    payload = response.json()
    models: list[OllamaRuntimeModel] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        raw_details = item.get("details")
        details: dict[str, object] = raw_details if isinstance(raw_details, dict) else {}
        context_window = item.get("context_length") or item.get("context")
        processor = item.get("processor")
        expires_at = item.get("expires_at")
        size_bytes = _optional_int(item.get("size"))
        size_vram_bytes = _optional_int(item.get("size_vram"))
        size_cpu_bytes = _size_cpu_bytes(size_bytes, size_vram_bytes)
        vram_percent = _vram_percent(size_bytes, size_vram_bytes)
        models.append(
            OllamaRuntimeModel(
                name=str(item.get("name") or item.get("model") or "unknown"),
                size_bytes=size_bytes,
                size_vram_bytes=size_vram_bytes,
                size_cpu_bytes=size_cpu_bytes,
                vram_percent=vram_percent,
                offload_status=_offload_status(size_bytes, size_vram_bytes),
                processor=(str(processor) if processor is not None else None),
                context_window=_optional_int(context_window or details.get("context_length")),
                expires_at=(str(expires_at) if expires_at is not None else None),
            )
        )
    return models


async def _run_game_job(
    *,
    job_id: str,
    jobs: dict[str, GameJob],
    tasks: dict[str, asyncio.Task[None]],
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    white: str,
    black: str,
    legality_mode: str,
    sampling: GameDefaults,
    ollama_thinking: bool,
    ollama_cpu_offload: bool,
    guidance_mode: GuidanceMode,
    max_plies: int | None,
    stockfish_path: str | None,
) -> None:
    try:
        def mark_game_started(game_id: int) -> None:
            jobs[job_id] = jobs[job_id].model_copy(update={"game_id": game_id})

        result, _attempt_log = await _play_async(
            settings=_settings_for_ollama_options(
                settings,
                sampling=sampling,
                thinking=ollama_thinking,
                cpu_offload=ollama_cpu_offload,
            ),
            db_url=settings.database_url,
            white_name=white,
            black_name=black,
            legality_mode=legality_mode,
            max_plies=max_plies,
            stockfish_path=stockfish_path,
            strategic_memory=guidance_mode == "strategic_memory",
            commit_after_each_ply=True,
            on_game_started=mark_game_started,
        )
    except asyncio.CancelledError:
        job = jobs.get(job_id)
        if job is not None:
            if job.game_id is not None:
                await _mark_game_cancelled(session_factory, game_id=job.game_id)
            jobs[job_id] = job.model_copy(
                update={
                    "status": "cancelled",
                    "termination_reason": "aborted_by_user",
                    "completed_at": job.completed_at or _utcnow_iso(),
                }
            )
        raise
    except Exception as exc:
        jobs[job_id] = jobs[job_id].model_copy(
            update={
                "status": "failed",
                "error": str(exc) or type(exc).__name__,
                "completed_at": _utcnow_iso(),
            }
        )
        return
    finally:
        tasks.pop(job_id, None)

    if jobs[job_id].status == "running":
        jobs[job_id] = jobs[job_id].model_copy(
            update={
                "status": "completed",
                "game_id": result.game_id,
                "game_ids": [result.game_id],
                "games_completed": 1,
                "result": result.result,
                "termination_reason": result.termination_reason,
                "completed_at": _utcnow_iso(),
            }
        )


async def _run_stockfish_match_job(
    *,
    job_id: str,
    jobs: dict[str, GameJob],
    tasks: dict[str, asyncio.Task[None]],
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    model: str,
    level: StockfishLevel,
    game_count: int,
    legality_mode: Literal["open", "constrained"],
    sampling: GameDefaults,
    ollama_thinking: bool,
    ollama_cpu_offload: bool,
    guidance_mode: GuidanceMode,
    max_plies: int | None,
    stockfish_path: str,
) -> None:
    try:
        await create_tables(settings.database_url)
        stockfish_settings = _settings_for_stockfish_level(
            _settings_for_ollama_options(
                settings,
                sampling=sampling,
                thinking=ollama_thinking,
                cpu_offload=ollama_cpu_offload,
            ),
            level=level,
            stockfish_path=stockfish_path,
        )
        evaluator = StockfishEvaluator(
            binary_path=stockfish_path,
            nodes=stockfish_settings.stockfish_nodes,
            threads=stockfish_settings.stockfish_threads,
            hash_mb=stockfish_settings.stockfish_hash_mb,
        )

        def mark_game_started(game_id: int, number: int) -> None:
            job = jobs.get(job_id)
            if job is None or job.status != "running":
                return
            jobs[job_id] = job.model_copy(
                update={
                    "game_id": game_id,
                    "games_completed": max(number - 1, 0),
                }
            )

        def mark_game_completed(game_id: int, completed: int) -> None:
            job = jobs.get(job_id)
            if job is None or job.status != "running":
                return
            game_ids = [*job.game_ids, game_id]
            jobs[job_id] = job.model_copy(
                update={
                    "game_id": game_id,
                    "game_ids": game_ids,
                    "games_completed": completed,
                }
            )

        async with session_factory() as session:
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name=f"{model} vs {_stockfish_display_name(level)}",
                    competitor_a=model,
                    competitor_b="stockfish",
                    legality_mode=legality_mode,
                    max_plies=max_plies,
                    game_count=game_count,
                    strategic_memory=guidance_mode == "strategic_memory",
                ),
                settings=stockfish_settings,
                source_factory=lambda source_name: _source_from_name(
                    source_name,
                    stockfish_settings,
                ),
                evaluator=evaluator,
                commit_after_each_ply=True,
                on_game_started=mark_game_started,
                on_game_completed=mark_game_completed,
            )
            jobs[job_id] = jobs[job_id].model_copy(update={"run_id": result.run_id})
            async with session.begin():
                summary_count = await rebuild_game_summaries(session, run_id=result.run_id)
                session.add(
                    OperationalEvent(
                        run_id=result.run_id,
                        event_kind="stockfish_match_summary",
                        severity="info",
                        message="Stockfish match summary materialized.",
                        payload={
                            "model": model,
                            "stockfish_level": level,
                            "games_requested": game_count,
                            "games_completed": len(result.game_ids),
                            "summary_rows": summary_count,
                        },
                    )
                )
    except asyncio.CancelledError:
        job = jobs.get(job_id)
        if job is not None:
            jobs[job_id] = job.model_copy(
                update={
                    "status": "cancelled",
                    "termination_reason": "aborted_by_user",
                    "completed_at": job.completed_at or _utcnow_iso(),
                }
            )
        raise
    except Exception as exc:
        jobs[job_id] = jobs[job_id].model_copy(
            update={
                "status": "failed",
                "error": str(exc) or type(exc).__name__,
                "completed_at": _utcnow_iso(),
            }
        )
        return
    finally:
        tasks.pop(job_id, None)

    if jobs[job_id].status == "running":
        jobs[job_id] = jobs[job_id].model_copy(
            update={
                "status": "completed",
                "run_id": result.run_id,
                "game_id": result.game_ids[-1] if result.game_ids else None,
                "game_ids": result.game_ids,
                "games_completed": len(result.game_ids),
                "result": "summary",
                "termination_reason": "completed",
                "completed_at": _utcnow_iso(),
            }
        )


async def _mark_game_cancelled(
    session_factory: async_sessionmaker[AsyncSession], *, game_id: int
) -> None:
    async with session_factory() as session:
        async with session.begin():
            game = await session.get(Game, game_id)
            if game is None or game.ended_at is not None:
                return
            game.result = "*"
            game.termination_reason = "aborted_by_user"
            game.ended_at = datetime.now(UTC)


def _effective_stockfish_path(settings: Settings, override: str | None) -> str | None:
    candidate = override or settings.stockfish_path
    if candidate:
        return candidate
    path_candidate = shutil.which("stockfish")
    if path_candidate:
        return path_candidate
    vendor_candidate = Path("vendor/stockfish/root/usr/games/stockfish")
    if vendor_candidate.exists():
        return str(vendor_candidate.resolve())
    return None


def _settings_for_stockfish_level(
    settings: Settings,
    *,
    level: StockfishLevel,
    stockfish_path: str,
) -> Settings:
    preset = _stockfish_level_options(level)
    return settings.model_copy(
        update={
            "stockfish_path": stockfish_path,
            "stockfish_skill": preset["skill"],
            "stockfish_limit_strength": True,
            "stockfish_target_elo": preset["target_elo"],
        }
    )


def _stockfish_level_options(level: StockfishLevel) -> dict[str, int]:
    if level == "club":
        return {"skill": 8, "target_elo": 1600}
    return {"skill": 2, "target_elo": 1320}


def _stockfish_display_name(level: StockfishLevel) -> str:
    preset = _stockfish_level_options(level)
    return f"Stockfish {preset['target_elo']} ({level})"


def _settings_for_ollama_options(
    settings: Settings,
    *,
    sampling: GameDefaults,
    thinking: bool,
    cpu_offload: bool,
) -> Settings:
    temperature = sampling.temperature
    top_p = sampling.top_p
    num_ctx = sampling.num_ctx
    num_predict = sampling.num_predict

    think = "off"
    if thinking:
        think = "auto"
        num_predict = max(num_predict or 0, 512)
        num_ctx = num_ctx or 32768

    num_gpu: int | None = None
    if cpu_offload:
        num_gpu = settings.ollama_cpu_offload_gpu_layers
        num_ctx = min(num_ctx or 8192, 8192)
        num_predict = num_predict or 256

    return settings.model_copy(
        update={
            "ollama_temperature": temperature,
            "ollama_top_p": top_p,
            "ollama_num_ctx": num_ctx,
            "ollama_num_predict": num_predict,
            "ollama_think": think,
            "ollama_num_gpu": num_gpu,
        }
    )


def _size_cpu_bytes(size_bytes: int | None, size_vram_bytes: int | None) -> int | None:
    if size_bytes is None or size_vram_bytes is None:
        return None
    return max(size_bytes - size_vram_bytes, 0)


def _vram_percent(size_bytes: int | None, size_vram_bytes: int | None) -> float | None:
    if size_bytes is None or size_vram_bytes is None or size_bytes <= 0:
        return None
    return min(max((size_vram_bytes / size_bytes) * 100, 0.0), 100.0)


def _offload_status(size_bytes: int | None, size_vram_bytes: int | None) -> str:
    vram_percent = _vram_percent(size_bytes, size_vram_bytes)
    if vram_percent is None:
        return "unknown"
    if vram_percent >= 98:
        return "gpu"
    if vram_percent <= 2:
        return "cpu"
    return "mixed"


async def _all_summaries(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[GameSummary]:
    async with session_factory() as session:
        rows = (await session.execute(select(GameSummary).order_by(GameSummary.run_id))).scalars()
        return list(rows.all())


async def _attempts_by_move(
    session_factory: async_sessionmaker[AsyncSession],
    game_id: int,
) -> dict[int, list[AttemptOut]]:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Attempt, TokenUsage)
                .outerjoin(TokenUsage, TokenUsage.attempt_id == Attempt.id)
                .where(Attempt.game_id == game_id, Attempt.move_id.is_not(None))
                .order_by(Attempt.ply, Attempt.attempt_number)
            )
        ).all()
    grouped: dict[int, list[AttemptOut]] = {}
    for attempt, token_usage in rows:
        if attempt.move_id is None:
            continue
        grouped.setdefault(attempt.move_id, []).append(
            AttemptOut(
                id=attempt.id,
                ply=attempt.ply,
                attempt_number=attempt.attempt_number,
                parsed_move=attempt.parsed_move,
                parse_ok=attempt.parse_ok,
                legal_ok=attempt.legal_ok,
                error_type=attempt.error_type,
                latency_ms=attempt.latency_ms,
                token_usage=_token_usage_out(token_usage),
            )
        )
    return grouped


async def _evaluations_by_move(
    session_factory: async_sessionmaker[AsyncSession],
    move_ids: list[int],
) -> dict[int, list[EngineEvaluationOut]]:
    if not move_ids:
        return {}
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(EngineEvaluation)
                .where(EngineEvaluation.move_id.in_(move_ids))
                .order_by(EngineEvaluation.id)
            )
        ).scalars()
    grouped: dict[int, list[EngineEvaluationOut]] = {}
    for row in rows:
        grouped.setdefault(row.move_id, []).append(
            EngineEvaluationOut(
                engine_name=row.engine_name,
                engine_version=row.engine_version,
                nodes=row.nodes,
                depth_reached=row.depth_reached,
                eval_before_cp=row.eval_before_cp,
                eval_after_cp=row.eval_after_cp,
                mate_before=row.mate_before,
                mate_after=row.mate_after,
                best_move_uci=row.best_move_uci,
                centipawn_loss=row.centipawn_loss,
                classification=row.classification,
            )
        )
    return grouped


async def _annotations_by_move(
    session_factory: async_sessionmaker[AsyncSession],
    move_ids: list[int],
) -> dict[int, list[MoveAnnotationOut]]:
    if not move_ids:
        return {}
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(MoveAnnotation)
                .where(MoveAnnotation.move_id.in_(move_ids))
                .order_by(MoveAnnotation.id)
            )
        ).scalars()
    grouped: dict[int, list[MoveAnnotationOut]] = {}
    for row in rows:
        grouped.setdefault(row.move_id, []).append(
            MoveAnnotationOut(
                persona=row.persona,
                commentary=row.commentary,
                created_at=row.created_at.isoformat(),
            )
        )
    return grouped


def _token_usage_out(row: TokenUsage | None) -> TokenUsageOut | None:
    if row is None:
        return None
    return TokenUsageOut(
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        estimated_context_window=row.estimated_context_window,
        estimated_context_remaining=row.estimated_context_remaining,
        truncation_applied=row.truncation_applied,
        cost_usd=row.cost_usd,
    )


def _operational_event_out(row: OperationalEvent) -> OperationalEventOut:
    return OperationalEventOut(
        id=row.id,
        run_id=row.run_id,
        event_kind=row.event_kind,
        severity=row.severity,
        message=row.message,
        payload=row.payload,
        created_at=row.created_at.isoformat(),
    )


def _comparison_row(run_id: int, rows: list[GameSummary]) -> RunComparisonRow:
    games_played = sum(row.games_played for row in rows)
    wins = sum(row.wins for row in rows)
    draws = sum(row.draws for row in rows)
    losses = sum(row.losses for row in rows)
    unfinished = sum(row.unfinished for row in rows)
    total_tokens = sum(row.total_tokens for row in rows)
    return RunComparisonRow(
        run_id=run_id,
        games_played=games_played,
        wins=wins,
        draws=draws,
        losses=losses,
        unfinished=unfinished,
        avg_cpl=_weighted_nullable_average([(row.avg_cpl, row.games_played) for row in rows]),
        avg_game_plies=_weighted_average(
            [(row.avg_game_plies, row.games_played) for row in rows]
        ),
        illegal_rate=_weighted_average([(row.illegal_rate, row.games_played) for row in rows]),
        malformed_rate=_weighted_average([(row.malformed_rate, row.games_played) for row in rows]),
        avg_retries=_weighted_average([(row.avg_retries, row.games_played) for row in rows]),
        avg_latency_ms=_weighted_average([(row.avg_latency_ms, row.games_played) for row in rows]),
        total_tokens=total_tokens,
    )


def _weighted_average(values: list[tuple[float, int]]) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight == 0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def _weighted_nullable_average(values: list[tuple[float | None, int]]) -> float | None:
    known = [(value, weight) for value, weight in values if value is not None]
    if not known:
        return None
    return _weighted_average([(value, weight) for value, weight in known])


def _pgn_header(pgn: str | None, tag: str) -> str | None:
    if not pgn:
        return None
    prefix = f'[{tag} "'
    for line in pgn.splitlines():
        if line.startswith(prefix) and line.endswith('"]'):
            value = line[len(prefix) : -2].strip()
            return value if value and value != "?" else None
    return None


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


app = create_app()

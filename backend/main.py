import asyncio
import json
import shutil
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arena_core.cli import _play_async
from arena_core.config import Settings, get_settings
from arena_core.persistence.database import create_session_factory
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

GameStreamPayload = dict[str, list[dict[str, int | str | None]]]
GameJobStatus = Literal["running", "completed", "failed", "cancelled"]
GuidanceMode = Literal["legal_list", "strategic_memory"]

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


class StartGameResponse(BaseModel):
    job_id: str
    status: GameJobStatus


class CancelGameResponse(BaseModel):
    job_id: str
    status: GameJobStatus
    game_id: int | None = None


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

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/models")
    async def list_models() -> list[ModelOption]:
        options = [ModelOption(id="random", label="random", provider="built-in")]
        if effective_settings.stockfish_path:
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
                "result": result.result,
                "termination_reason": result.termination_reason,
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

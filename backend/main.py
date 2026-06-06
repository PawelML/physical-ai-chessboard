import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from arena_core.config import Settings, get_settings
from arena_core.persistence.database import create_session_factory
from arena_core.persistence.models import (
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

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

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
            return GameDetail(
                id=game.id,
                run_id=game.run_id,
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
    total_tokens = sum(row.total_tokens for row in rows)
    return RunComparisonRow(
        run_id=run_id,
        games_played=games_played,
        wins=wins,
        draws=draws,
        losses=losses,
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


app = create_app()

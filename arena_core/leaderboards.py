from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from arena_core.persistence import models


@dataclass
class _Bucket:
    participant: models.RunParticipant
    color: str
    mode: str
    legality_mode: str
    opening_suite_id: int | None
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    unfinished: int = 0
    cpls: list[int] = field(default_factory=list)
    blunders: int = 0
    mistakes: int = 0
    inaccuracies: int = 0
    attempts: int = 0
    illegal_attempts: int = 0
    malformed_attempts: int = 0
    retries: list[int] = field(default_factory=list)
    forfeit_invalid_count: int = 0
    latencies: list[float] = field(default_factory=list)
    total_tokens: int = 0


async def rebuild_game_summaries(session: AsyncSession, *, run_id: int | None = None) -> int:
    if run_id is None:
        await session.execute(delete(models.GameSummary))
    else:
        await session.execute(delete(models.GameSummary).where(models.GameSummary.run_id == run_id))

    games_query = (
        select(models.Game)
        .where(models.Game.run_id.is_not(None))
        .order_by(models.Game.id)
    )
    if run_id is not None:
        games_query = games_query.where(models.Game.run_id == run_id)
    games = (await session.execute(games_query)).scalars().all()
    buckets: dict[tuple[int, str], _Bucket] = {}
    for game in games:
        if game.run_id is None:
            continue
        run = await session.get(models.BenchmarkRun, game.run_id)
        if run is None:
            continue
        prompt = await session.get(models.Prompt, run.prompt_id) if run.prompt_id else None
        mode = prompt.mode if prompt else "strict"
        legality_mode = prompt.legality_mode if prompt else "open"
        for color, participant_id in (
            ("white", game.white_participant_id),
            ("black", game.black_participant_id),
        ):
            if participant_id is None:
                continue
            participant = await session.get(models.RunParticipant, participant_id)
            if participant is None:
                continue
            key = (participant.id, color)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    participant=participant,
                    color=color,
                    mode=mode,
                    legality_mode=legality_mode,
                    opening_suite_id=run.opening_suite_id,
                )
                buckets[key] = bucket
            await _accumulate_game(session, bucket, game)

    for bucket in buckets.values():
        session.add(_summary_row(bucket))
    await session.flush()
    return len(buckets)


async def _accumulate_game(
    session: AsyncSession,
    bucket: _Bucket,
    game: models.Game,
) -> None:
    bucket.games_played += 1
    outcome = _outcome_for_color(game.result, bucket.color)
    if outcome == "win":
        bucket.wins += 1
    elif outcome == "loss":
        bucket.losses += 1
    elif outcome == "draw":
        bucket.draws += 1
    elif outcome == "unfinished":
        bucket.unfinished += 1
    if game.termination_reason == "forfeit_invalid" and outcome == "loss":
        bucket.forfeit_invalid_count += 1

    moves = (
        await session.execute(
            select(models.Move).where(
                models.Move.game_id == game.id,
                models.Move.color == bucket.color,
            )
        )
    ).scalars().all()
    move_ids = [move.id for move in moves]
    bucket.retries.extend(move.retries_used for move in moves)
    bucket.latencies.extend(move.latency_total_ms for move in moves)
    if move_ids:
        evaluations = (
            await session.execute(
                select(models.EngineEvaluation).where(models.EngineEvaluation.move_id.in_(move_ids))
            )
        ).scalars()
        for evaluation in evaluations:
            if evaluation.centipawn_loss is not None:
                bucket.cpls.append(evaluation.centipawn_loss)
            if evaluation.classification == "blunder":
                bucket.blunders += 1
            elif evaluation.classification == "mistake":
                bucket.mistakes += 1
            elif evaluation.classification == "inaccuracy":
                bucket.inaccuracies += 1

    attempts = (
        await session.execute(
            select(models.Attempt, models.TokenUsage)
            .outerjoin(models.TokenUsage, models.TokenUsage.attempt_id == models.Attempt.id)
            .where(models.Attempt.game_id == game.id)
        )
    ).all()
    for attempt, token_usage in attempts:
        if _color_for_ply(attempt.ply) != bucket.color:
            continue
        bucket.attempts += 1
        if not attempt.parse_ok:
            bucket.malformed_attempts += 1
        elif not attempt.legal_ok:
            bucket.illegal_attempts += 1
        if token_usage is not None:
            bucket.total_tokens += token_usage.total_tokens


def _summary_row(bucket: _Bucket) -> models.GameSummary:
    return models.GameSummary(
        run_id=bucket.participant.run_id,
        run_participant_id=bucket.participant.id,
        model_snapshot_id=bucket.participant.model_snapshot_id,
        color=bucket.color,
        mode=bucket.mode,
        legality_mode=bucket.legality_mode,
        opening_suite_id=bucket.opening_suite_id,
        games_played=bucket.games_played,
        wins=bucket.wins,
        draws=bucket.draws,
        losses=bucket.losses,
        unfinished=bucket.unfinished,
        avg_cpl=_avg(bucket.cpls),
        blunders=bucket.blunders,
        mistakes=bucket.mistakes,
        inaccuracies=bucket.inaccuracies,
        illegal_rate=_rate(bucket.illegal_attempts, bucket.attempts),
        malformed_rate=_rate(bucket.malformed_attempts, bucket.attempts),
        avg_retries=_avg(bucket.retries) or 0.0,
        forfeit_invalid_count=bucket.forfeit_invalid_count,
        avg_latency_ms=_avg(bucket.latencies) or 0.0,
        total_tokens=bucket.total_tokens,
    )


def _outcome_for_color(result: str, color: str) -> str:
    if result == "*":
        return "unfinished"
    if result == "1/2-1/2":
        return "draw"
    if result == "1-0":
        return "win" if color == "white" else "loss"
    if result == "0-1":
        return "win" if color == "black" else "loss"
    return "draw"


def _color_for_ply(ply: int) -> str:
    return "white" if ply % 2 == 1 else "black"


def _avg(values: list[int] | list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total

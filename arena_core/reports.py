from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arena_core.persistence import models


async def export_game_report(session: AsyncSession, *, game_id: int) -> str:
    game = await session.get(models.Game, game_id)
    if game is None:
        raise ValueError(f"Game {game_id} not found")
    moves = (
        await session.execute(
            select(models.Move).where(models.Move.game_id == game_id).order_by(models.Move.ply)
        )
    ).scalars().all()
    lines = [
        f"# Match Report — game {game.id}",
        "",
        f"Result: {game.result}",
        f"Termination: {game.termination_reason or 'unknown'}",
        f"Final FEN: `{game.final_fen or ''}`",
        "",
        "## PGN",
        "",
        "```pgn",
        game.pgn or "",
        "```",
        "",
        "## Move Timeline",
        "",
        "| Ply | Move | Eval | CPL | Retries | Latency | Commentary |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for move in moves:
        evaluation = await _evaluation_for_move(session, move.id)
        annotations = await _annotations_for_move(session, move.id)
        commentary = "<br>".join(f"**{row.persona}:** {row.commentary}" for row in annotations)
        cpl = (
            evaluation.centipawn_loss
            if evaluation and evaluation.centipawn_loss is not None
            else "—"
        )
        lines.append(
            "| "
            f"{move.ply} | "
            f"{move.accepted_san} `{move.accepted_uci}` | "
            f"{evaluation.classification if evaluation else '—'} | "
            f"{cpl} | "
            f"{move.retries_used} | "
            f"{move.latency_total_ms:.1f} ms | "
            f"{commentary or '—'} |"
        )
    return "\n".join(lines) + "\n"


async def _evaluation_for_move(
    session: AsyncSession,
    move_id: int,
) -> models.EngineEvaluation | None:
    return (
        await session.execute(
            select(models.EngineEvaluation).where(models.EngineEvaluation.move_id == move_id)
        )
    ).scalar_one_or_none()


async def _annotations_for_move(
    session: AsyncSession,
    move_id: int,
) -> list[models.MoveAnnotation]:
    result = await session.execute(
        select(models.MoveAnnotation)
        .where(models.MoveAnnotation.move_id == move_id)
        .order_by(models.MoveAnnotation.id)
    )
    return list(result.scalars().all())

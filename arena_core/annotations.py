from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arena_core.persistence import models
from arena_core.persistence.repositories import ensure_prompt, text_hash
from arena_core.prompts import LegalityMode, build_reasoning_prompt

PERSONAS = {"aggressive", "positional", "defensive", "risk-taking", "technician"}


async def annotate_game(
    session: AsyncSession,
    *,
    game_id: int,
    persona: str,
    legality_mode: LegalityMode = "open",
) -> int:
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    moves = (
        await session.execute(
            select(models.Move).where(models.Move.game_id == game_id).order_by(models.Move.ply)
        )
    ).scalars().all()
    count = 0
    for move in moves:
        existing = (
            await session.execute(
                select(models.MoveAnnotation).where(
                    models.MoveAnnotation.move_id == move.id,
                    models.MoveAnnotation.persona == persona,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        evaluation = (
            await session.execute(
                select(models.EngineEvaluation).where(models.EngineEvaluation.move_id == move.id)
            )
        ).scalar_one_or_none()
        evaluation_label = evaluation.classification if evaluation else "not evaluated"
        prompt = build_reasoning_prompt(
            fen_before=move.fen_before,
            accepted_san=move.accepted_san,
            accepted_uci=move.accepted_uci,
            persona=persona,
            evaluation_label=evaluation_label,
            legality_mode=legality_mode,
        )
        prompt_row = await ensure_prompt(session, prompt)
        commentary = _commentary(move, persona, evaluation_label)
        session.add(
            models.MoveAnnotation(
                move_id=move.id,
                prompt_id=prompt_row.id,
                persona=persona,
                commentary=commentary,
                raw_prompt_hash=text_hash(prompt.text),
                raw_prompt=prompt.text,
                raw_response=commentary,
            )
        )
        count += 1
    await session.flush()
    return count


def _commentary(move: models.Move, persona: str, evaluation_label: str) -> str:
    prefix = {
        "aggressive": "Presses for activity",
        "positional": "Keeps the structure in view",
        "defensive": "Prioritizes stability",
        "risk-taking": "Accepts practical imbalance",
        "technician": "Focuses on conversion details",
    }[persona]
    return (
        f"{prefix}: {move.accepted_san} ({move.accepted_uci}) was recorded as "
        f"{evaluation_label}; retries used: {move.retries_used}."
    )

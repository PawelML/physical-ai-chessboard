from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arena_core.persistence import models
from arena_core.prompts import BuiltPrompt


async def ensure_prompt(session: AsyncSession, prompt: BuiltPrompt) -> models.Prompt:
    result = await session.execute(
        select(models.Prompt).where(
            models.Prompt.version == prompt.version,
            models.Prompt.template_hash == prompt.template_hash,
            models.Prompt.mode == prompt.mode,
            models.Prompt.legality_mode == prompt.legality_mode,
        )
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        return existing
    row = models.Prompt(
        version=prompt.version,
        template_hash=prompt.template_hash,
        mode=prompt.mode,
        legality_mode=prompt.legality_mode,
    )
    session.add(row)
    await session.flush()
    return row

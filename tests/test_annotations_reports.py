from pathlib import Path

import pytest
from sqlalchemy import select

from arena_core.annotations import annotate_game
from arena_core.config import Settings
from arena_core.engine import ArenaGame, RandomMoveSource, StaticMoveSource
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.persistence.models import MoveAnnotation
from arena_core.reports import export_game_report


@pytest.mark.integration
async def test_annotate_game_persists_persona_commentary(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await ArenaGame(
                white=StaticMoveSource(['{"move":"e2e4"}']),
                black=RandomMoveSource(),
                settings=Settings(max_retries=0),
                max_plies=1,
            ).run(session)
            count = await annotate_game(session, game_id=result.game_id, persona="technician")

    async with session_factory() as session:
        annotation = (await session.execute(select(MoveAnnotation))).scalar_one()

    assert count == 1
    assert annotation.persona == "technician"
    assert "e4" in annotation.commentary


@pytest.mark.integration
async def test_export_game_report_includes_pgn_and_commentary(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await ArenaGame(
                white=StaticMoveSource(['{"move":"e2e4"}']),
                black=RandomMoveSource(),
                settings=Settings(max_retries=0),
                max_plies=1,
            ).run(session)
            await annotate_game(session, game_id=result.game_id, persona="aggressive")
            report = await export_game_report(session, game_id=result.game_id)

    assert "# Match Report" in report
    assert "```pgn" in report
    assert "**aggressive:**" in report

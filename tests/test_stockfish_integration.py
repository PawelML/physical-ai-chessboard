from pathlib import Path

import pytest
from sqlalchemy import func, select

from arena_core.config import Settings
from arena_core.engine import ArenaGame, RandomMoveSource
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.persistence.models import EngineEvaluation, GameSummary, Move
from arena_core.tournaments import TournamentConfig, run_tournament

STOCKFISH_PATH = Path("vendor/stockfish/root/usr/games/stockfish")


pytestmark = pytest.mark.integration


def _stockfish_path() -> str:
    if not STOCKFISH_PATH.exists():
        pytest.skip("vendor Stockfish binary is not available")
    return str(STOCKFISH_PATH.resolve())


async def test_real_stockfish_evaluator_persists_engine_rows(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    stockfish_path = _stockfish_path()
    settings = Settings(stockfish_path=stockfish_path, stockfish_nodes=10_000)

    from arena_core.evaluators.stockfish import StockfishEvaluator

    async with session_factory() as session:
        async with session.begin():
            await ArenaGame(
                white=RandomMoveSource(),
                black=RandomMoveSource(),
                settings=settings,
                max_plies=1,
                evaluator=StockfishEvaluator(
                    binary_path=stockfish_path,
                    nodes=settings.stockfish_nodes,
                ),
            ).run(session)

    async with session_factory() as session:
        evaluation = (await session.execute(select(EngineEvaluation))).scalar_one()

    assert evaluation.engine_name == "stockfish"
    assert "Stockfish 16" in evaluation.engine_version
    assert evaluation.nodes == 10_000
    assert evaluation.best_move_uci is not None


async def test_real_stockfish_source_runs_tournament_game(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    settings = Settings(
        stockfish_path=_stockfish_path(),
        stockfish_nodes=10_000,
        stockfish_skill=2,
        stockfish_limit_strength=True,
        stockfish_target_elo=1400,
    )

    from arena_core.cli import _source_from_name
    from arena_core.leaderboards import rebuild_game_summaries

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="real-stockfish-smoke",
                    competitor_a="stockfish",
                    competitor_b="random",
                    max_plies=1,
                ),
                settings=settings,
                source_factory=lambda source_name, _rng: _source_from_name(source_name, settings),
            )
            await rebuild_game_summaries(session, run_id=result.run_id)

    async with session_factory() as session:
        move_count = await session.scalar(select(func.count()).select_from(Move))
        summary_count = await session.scalar(select(func.count()).select_from(GameSummary))

    assert move_count == 20
    assert summary_count == 4

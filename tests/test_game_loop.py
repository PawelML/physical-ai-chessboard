from pathlib import Path

import chess
import pytest
from sqlalchemy import func, select

from arena_core.config import Settings
from arena_core.engine import ArenaGame, RandomMoveSource, StaticMoveSource
from arena_core.evaluators.stockfish import EngineEvaluation
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.persistence.models import Attempt, Move, TokenUsage
from arena_core.persistence.models import EngineEvaluation as EngineEvaluationRow


class FakeEvaluator:
    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        return EngineEvaluation(
            engine_name="fakefish",
            engine_version="fakefish 1",
            nodes=1,
            depth_reached=1,
            eval_before_cp=20,
            eval_after_cp=10,
            mate_before=None,
            mate_after=None,
            best_move_uci=move.uci(),
            centipawn_loss=10,
            classification="best",
        )


@pytest.mark.integration
async def test_random_agents_persist_one_move(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await ArenaGame(
                white=RandomMoveSource(),
                black=RandomMoveSource(),
                settings=Settings(max_retries=3),
                max_plies=1,
            ).run(session)

    async with session_factory() as session:
        move_count = await session.scalar(select(func.count()).select_from(Move))
        attempt_count = await session.scalar(select(func.count()).select_from(Attempt))
        token_count = await session.scalar(select(func.count()).select_from(TokenUsage))

    assert result.plies == 1
    assert result.termination_reason == "max_plies"
    assert move_count == 1
    assert attempt_count == 1
    assert token_count == 1


@pytest.mark.integration
async def test_invalid_attempts_are_persisted_and_retried(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    white = StaticMoveSource(["not json", '{"move":"e2e5"}', '{"move":"e2e4"}'])

    async with session_factory() as session:
        async with session.begin():
            result = await ArenaGame(
                white=white,
                black=RandomMoveSource(),
                settings=Settings(max_retries=3),
                max_plies=1,
            ).run(session)

    async with session_factory() as session:
        attempts = (
            await session.execute(select(Attempt).order_by(Attempt.attempt_number))
        ).scalars().all()
        move = (await session.execute(select(Move))).scalar_one()

    assert result.plies == 1
    assert move.accepted_uci == "e2e4"
    assert move.retries_used == 2
    assert [attempt.legal_ok for attempt in attempts] == [False, False, True]
    assert attempts[0].error_type == "malformed_json"
    assert attempts[1].error_type == "illegal_move"
    assert attempts[2].move_id == move.id


@pytest.mark.integration
async def test_engine_evaluation_is_persisted_for_accepted_move(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            await ArenaGame(
                white=StaticMoveSource(['{"move":"e2e4"}']),
                black=RandomMoveSource(),
                settings=Settings(max_retries=0),
                max_plies=1,
                evaluator=FakeEvaluator(),
            ).run(session)

    async with session_factory() as session:
        evaluation = (await session.execute(select(EngineEvaluationRow))).scalar_one()

    assert evaluation.engine_name == "fakefish"
    assert evaluation.best_move_uci == "e2e4"
    assert evaluation.centipawn_loss == 10

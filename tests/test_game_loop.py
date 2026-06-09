from pathlib import Path

import chess
import pytest
from sqlalchemy import func, select

from arena_core.config import Settings
from arena_core.engine import ArenaGame, RandomMoveSource, StaticMoveSource
from arena_core.evaluators.stockfish import EngineEvaluation
from arena_core.leaderboards import _outcome_for_color, rebuild_game_summaries
from arena_core.move_sources import MoveProposal
from arena_core.persistence import models
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.persistence.models import Attempt, GameSummary, Move, TokenUsage
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


class FailingMoveSource:
    name = "failing"
    source_type = "llm"

    async def propose(self, *, prompt: str, board: chess.Board) -> MoveProposal:
        raise RuntimeError("model failed")


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
async def test_commit_after_each_ply_marks_partial_game_failed_on_exception(
    tmp_path: Path,
) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        with pytest.raises(RuntimeError, match="model failed"):
            await ArenaGame(
                white=StaticMoveSource(['{"move":"e2e4"}']),
                black=FailingMoveSource(),
                settings=Settings(max_retries=0),
            ).run(session, commit_after_each_ply=True)

    async with session_factory() as session:
        game = (await session.execute(select(models.Game))).scalar_one()
        moves = (await session.execute(select(Move).order_by(Move.ply))).scalars().all()

    assert game.result == "*"
    assert game.termination_reason == "error"
    assert game.ended_at is not None
    assert game.final_fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    assert [move.accepted_uci for move in moves] == ["e2e4"]


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
async def test_only_uci_moves_are_accepted(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    white = StaticMoveSource(['{"move":"e4"}', '{"move":"e2e4"}'])
    black = StaticMoveSource(['{"move":"e7e5"}'])

    async with session_factory() as session:
        async with session.begin():
            result = await ArenaGame(
                white=white,
                black=black,
                settings=Settings(max_retries=1),
                max_plies=2,
            ).run(session)

    async with session_factory() as session:
        moves = (
            await session.execute(select(Move).order_by(Move.ply))
        ).scalars().all()
        attempts = (
            await session.execute(select(Attempt).order_by(Attempt.ply, Attempt.attempt_number))
        ).scalars().all()

    assert result.plies == 2
    assert [(m.accepted_san, m.accepted_uci) for m in moves] == [
        ("e4", "e2e4"),
        ("e5", "e7e5"),
    ]
    assert [attempt.legal_ok for attempt in attempts] == [False, True, True]
    assert attempts[0].parsed_move == "e4"
    assert attempts[0].error_type == "illegal_move"
    assert moves[0].retries_used == 1


@pytest.mark.integration
async def test_forfeit_by_invalid_scores_loss_for_forfeiting_side(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    white = StaticMoveSource(['{"move":"e2e5"}', '{"move":"e2e5"}'])

    async with session_factory() as session:
        async with session.begin():
            run = models.BenchmarkRun(
                name="invalid-forfeit",
                config_hash="f" * 64,
            )
            session.add(run)
            await session.flush()
            white_participant = models.RunParticipant(
                run_id=run.id,
                opponent_type="model",
                color_policy="white",
                display_name="illegal-white",
            )
            black_participant = models.RunParticipant(
                run_id=run.id,
                opponent_type="random",
                color_policy="black",
                display_name="random-black",
            )
            session.add_all([white_participant, black_participant])
            await session.flush()
            result = await ArenaGame(
                white=white,
                black=RandomMoveSource(),
                settings=Settings(max_retries=1),
                run_id=run.id,
                white_participant_id=white_participant.id,
                black_participant_id=black_participant.id,
            ).run(session)
            await rebuild_game_summaries(session, run_id=run.id)

    async with session_factory() as session:
        summaries = (await session.execute(select(GameSummary))).scalars().all()

    assert result.termination_reason == "forfeit_invalid"
    assert result.result == "0-1"
    assert _outcome_for_color(result.result, "white") == "loss"
    assert _outcome_for_color(result.result, "black") == "win"
    by_color = {summary.color: summary for summary in summaries}
    assert by_color["white"].losses == 1
    assert by_color["white"].forfeit_invalid_count == 1
    assert by_color["black"].wins == 1
    assert by_color["black"].forfeit_invalid_count == 0


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

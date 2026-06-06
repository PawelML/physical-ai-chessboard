from pathlib import Path

import chess
import pytest
from sqlalchemy import func, select

from arena_core.config import Settings
from arena_core.engine import RandomMoveSource
from arena_core.evaluators.stockfish import EngineEvaluation
from arena_core.leaderboards import rebuild_game_summaries
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.persistence.models import (
    BenchmarkRun,
    Game,
    GameSummary,
    Model,
    ModelSnapshot,
    OpeningLine,
    OperationalEvent,
    RunParticipant,
)
from arena_core.tournaments import TournamentConfig, board_from_move_sequence, run_tournament


class FakeVersionedEvaluator:
    version = "Fakefish 1.0"

    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        raise NotImplementedError


@pytest.mark.integration
async def test_tournament_persists_run_openings_and_both_colors(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="smoke",
                    competitor_a="random",
                    competitor_b="random",
                    max_plies=1,
                    seed=7,
                ),
                settings=Settings(max_retries=0),
                source_factory=lambda _name: RandomMoveSource(),
            )

    async with session_factory() as session:
        run = await session.get(BenchmarkRun, result.run_id)
        participant_count = await session.scalar(select(func.count()).select_from(RunParticipant))
        opening_count = await session.scalar(select(func.count()).select_from(OpeningLine))
        games = (await session.execute(select(Game).order_by(Game.id))).scalars().all()

    assert run is not None
    assert len(result.config_hash) == 64
    assert participant_count == 2
    assert opening_count == 2
    assert len(games) == 4
    assert {game.white_participant_id for game in games} == {
        games[0].white_participant_id,
        games[1].white_participant_id,
    }
    assert all(game.run_id == result.run_id for game in games)
    assert all(game.opening_line_id is not None for game in games)
    assert all(game.termination_reason == "max_plies" for game in games)


def test_board_from_move_sequence_applies_opening() -> None:
    board = board_from_move_sequence("e2e4 e7e5")

    assert board.ply() == 2
    assert board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3")


@pytest.mark.integration
async def test_stockfish_participant_metadata_is_captured_without_playing(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="stockfish-metadata",
                    competitor_a="stockfish",
                    competitor_b="random",
                    max_plies=0,
                ),
                settings=Settings(
                    max_retries=0,
                    stockfish_path="/usr/bin/stockfish",
                    stockfish_skill=3,
                    stockfish_limit_strength=True,
                    stockfish_target_elo=1400,
                ),
                source_factory=lambda _name: RandomMoveSource(),
            )

    async with session_factory() as session:
        participants = (
            await session.execute(
                select(RunParticipant).where(RunParticipant.run_id == result.run_id)
            )
        ).scalars().all()

    stockfish = next(row for row in participants if row.display_name == "stockfish")
    assert stockfish.opponent_type == "stockfish"
    assert stockfish.stockfish_skill == 3
    assert stockfish.uci_limit_strength is True
    assert stockfish.target_elo == 1400


@pytest.mark.integration
async def test_tournament_records_evaluator_stockfish_version(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="stockfish-version",
                    competitor_a="random",
                    competitor_b="random",
                    max_plies=0,
                ),
                settings=Settings(max_retries=0),
                source_factory=lambda _name: RandomMoveSource(),
                evaluator=FakeVersionedEvaluator(),
            )

    async with session_factory() as session:
        run = await session.get(BenchmarkRun, result.run_id)

    assert run is not None
    assert run.stockfish_version == "Fakefish 1.0"


@pytest.mark.integration
async def test_tournament_records_model_pair_vram_warning(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="vram-warning",
                    competitor_a="qwen3.5:35b-a3b",
                    competitor_b="qwen3.5-27b-ud",
                    max_plies=0,
                ),
                settings=Settings(max_retries=0, ollama_vram_budget_gb=24),
                source_factory=lambda _name: RandomMoveSource(),
            )

    async with session_factory() as session:
        event = (
            await session.execute(
                select(OperationalEvent).where(OperationalEvent.run_id == result.run_id)
            )
        ).scalar_one()

    assert event.event_kind == "model_swap_warning"
    assert event.severity == "warning"
    assert event.payload is not None
    assert event.payload["total_vram_gb"] == 40.0


@pytest.mark.integration
async def test_tournament_captures_ollama_model_snapshot_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async def fake_metadata(
        *,
        model: str,
        base_url: str,
        timeout_seconds: float,
    ) -> object:
        from arena_core.llm.ollama import OllamaModelMetadata

        return OllamaModelMetadata(
            name=model,
            digest=f"digest-{model}",
            family="qwen35",
            parameter_size="9.7B",
            quantization="Q4_K_M",
            context_window=262144,
            runtime_version="0.17.7",
            modified_at="2026-05-27T17:29:40+02:00",
            size_bytes=6594474711,
        )

    monkeypatch.setattr("arena_core.tournaments.fetch_ollama_model_metadata", fake_metadata)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="model-snapshots",
                    competitor_a="qwen3.5:9b",
                    competitor_b="gemma3n:e4b",
                    max_plies=0,
                ),
                settings=Settings(max_retries=0),
                source_factory=lambda _name: RandomMoveSource(),
            )

    async with session_factory() as session:
        run = await session.get(BenchmarkRun, result.run_id)
        rows = (
            await session.execute(
                select(RunParticipant, ModelSnapshot, Model)
                .join(ModelSnapshot, RunParticipant.model_snapshot_id == ModelSnapshot.id)
                .join(Model, ModelSnapshot.model_id == Model.id)
                .where(RunParticipant.run_id == result.run_id)
            )
        ).all()

    assert run is not None
    assert len(run.config_hash) == 64
    assert len(rows) == 2
    for _participant, snapshot, model in rows:
        assert model.provider == "local"
        assert model.family == "qwen35"
        assert model.param_size == "9.7B"
        assert snapshot.ollama_digest == f"digest-{model.name}"
        assert snapshot.quantization == "Q4_K_M"
        assert snapshot.context_window == 262144
        assert snapshot.runtime_version == "0.17.7"
        assert snapshot.sampler_params == {"temperature": 0, "format": "json", "think": False}


@pytest.mark.integration
async def test_rebuild_game_summaries_materializes_leaderboard_rows(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="summary-smoke",
                    competitor_a="random",
                    competitor_b="random",
                    max_plies=1,
                ),
                settings=Settings(max_retries=0),
                source_factory=lambda _name: RandomMoveSource(),
            )
            row_count = await rebuild_game_summaries(session, run_id=result.run_id)

    async with session_factory() as session:
        summaries = (await session.execute(select(GameSummary))).scalars().all()

    assert row_count == 4
    assert len(summaries) == 4
    assert sum(summary.games_played for summary in summaries) == 8
    assert {summary.games_played for summary in summaries} == {2}
    assert {summary.unfinished for summary in summaries} == {2}
    assert all(summary.draws == 0 for summary in summaries)
    assert all(summary.legality_mode == "open" for summary in summaries)
    assert sum(summary.total_tokens for summary in summaries) > 0

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
                source_factory=lambda _name, _rng: RandomMoveSource(),
            )

    async with session_factory() as session:
        run = await session.get(BenchmarkRun, result.run_id)
        participant_count = await session.scalar(select(func.count()).select_from(RunParticipant))
        opening_count = await session.scalar(select(func.count()).select_from(OpeningLine))
        games = (await session.execute(select(Game).order_by(Game.id))).scalars().all()

    assert run is not None
    assert len(result.config_hash) == 64
    assert participant_count == 2
    assert opening_count == 10
    assert len(games) == 20
    assert {game.white_participant_id for game in games} == {
        games[0].white_participant_id,
        games[1].white_participant_id,
    }
    assert all(game.run_id == result.run_id for game in games)
    assert all(game.opening_line_id is not None for game in games)
    assert all(game.termination_reason == "max_plies" for game in games)


@pytest.mark.integration
async def test_tournament_can_limit_exact_game_count(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="limited-count",
                    competitor_a="random",
                    competitor_b="random",
                    max_plies=1,
                    game_count=3,
                ),
                settings=Settings(max_retries=0),
                source_factory=lambda _name, _rng: RandomMoveSource(),
            )

    async with session_factory() as session:
        games = (
            await session.execute(
                select(Game).where(Game.run_id == result.run_id).order_by(Game.id)
            )
        ).scalars().all()

    assert len(result.game_ids) == 3
    assert len(games) == 3
    assert len({game.opening_line_id for game in games}) == 2


@pytest.mark.integration
async def test_tournament_seed_makes_random_games_deterministic(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    config = TournamentConfig(
        name="deterministic",
        competitor_a="random",
        competitor_b="random",
        max_plies=4,
        game_count=4,
        seed=42,
    )

    async with session_factory() as session:
        async with session.begin():
            first = await run_tournament(
                session=session,
                config=config,
                settings=Settings(max_retries=0),
                source_factory=lambda _name, rng: RandomMoveSource(rng=rng),
            )
            second = await run_tournament(
                session=session,
                config=config,
                settings=Settings(max_retries=0),
                source_factory=lambda _name, rng: RandomMoveSource(rng=rng),
            )

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Game.run_id, Game.pgn)
                .where(Game.run_id.in_([first.run_id, second.run_id]))
                .order_by(Game.run_id, Game.id)
            )
        ).all()

    first_pgns = [pgn for run_id, pgn in rows if run_id == first.run_id]
    second_pgns = [pgn for run_id, pgn in rows if run_id == second.run_id]
    assert first_pgns == second_pgns


@pytest.mark.integration
async def test_tournament_live_commit_exposes_started_game(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)
    visible_started_games: list[int] = []

    async def record_visible_game(game_id: int, _game_number: int) -> None:
        async with session_factory() as observer:
            visible = await observer.get(Game, game_id)
        if visible is not None:
            visible_started_games.append(game_id)

    async with session_factory() as session:
        result = await run_tournament(
            session=session,
            config=TournamentConfig(
                name="live-commit",
                competitor_a="random",
                competitor_b="random",
                max_plies=1,
                game_count=1,
            ),
            settings=Settings(max_retries=0),
            source_factory=lambda _name, _rng: RandomMoveSource(),
            commit_after_each_ply=True,
            on_game_started=record_visible_game,
        )

    async with session_factory() as session:
        game = await session.get(Game, result.game_ids[0])

    assert visible_started_games == result.game_ids
    assert game is not None
    assert game.ended_at is not None


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
                source_factory=lambda _name, _rng: RandomMoveSource(),
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
                source_factory=lambda _name, _rng: RandomMoveSource(),
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
                source_factory=lambda _name, _rng: RandomMoveSource(),
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
                settings=Settings(
                    max_retries=0,
                    ollama_temperature=0.2,
                    ollama_top_p=0.9,
                    ollama_num_ctx=32768,
                    ollama_num_predict=256,
                    ollama_num_gpu=32,
                    ollama_think="auto",
                ),
                source_factory=lambda _name, _rng: RandomMoveSource(),
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
        assert snapshot.context_window == 32768
        assert snapshot.runtime_version == "0.17.7"
        assert snapshot.sampler_params == {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_ctx": 32768,
            "num_predict": 256,
            "num_gpu": 32,
            "cpu_offload_gpu_layers": 48,
            "cpu_offload_min_gpu_layers": 8,
            "format": "json",
            "think": "auto",
        }


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
                source_factory=lambda _name, _rng: RandomMoveSource(),
            )
            row_count = await rebuild_game_summaries(session, run_id=result.run_id)

    async with session_factory() as session:
        summaries = (await session.execute(select(GameSummary))).scalars().all()

    assert row_count == 4
    assert len(summaries) == 4
    assert sum(summary.games_played for summary in summaries) == 40
    assert {summary.games_played for summary in summaries} == {10}
    assert {summary.unfinished for summary in summaries} == {10}
    assert {summary.avg_game_plies for summary in summaries} == {1.0}
    assert all(summary.draws == 0 for summary in summaries)
    assert all(summary.legality_mode == "open" for summary in summaries)
    assert all(summary.low_sample is False for summary in summaries)
    assert all(summary.attempt_count >= 0 for summary in summaries)
    assert sum(summary.total_tokens for summary in summaries) > 0

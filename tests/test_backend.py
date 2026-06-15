from pathlib import Path

from fastapi.testclient import TestClient

from arena_core.annotations import annotate_game
from arena_core.config import Settings
from arena_core.engine import ArenaGame, RandomMoveSource, StaticMoveSource
from arena_core.leaderboards import rebuild_game_summaries
from arena_core.llm.base import LLMResponse, LLMService
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.tournaments import TournamentConfig, run_tournament
from backend.main import (
    GameDefaults,
    _game_stream_payload,
    _gemini_model_options,
    _settings_for_ollama_options,
    _settings_for_stockfish_level,
    create_app,
)


class StubLLMService(LLMService):
    async def complete(self, *, model: str, prompt: str) -> LLMResponse:
        return LLMResponse(content="backend commentary")


def test_gemini_model_option_uses_configured_model() -> None:
    options = _gemini_model_options(
        Settings(
            api_providers_enabled=True,
            gemini_api_key="test-key",
            gemini_model="gemini-3-flash-preview",
        )
    )

    assert [option.model_dump() for option in options] == [
        {
            "id": "gemini:gemini-3-flash-preview",
            "label": "gemini-3-flash-preview",
            "provider": "gemini",
        }
    ]


def test_gemini_model_option_requires_enabled_provider() -> None:
    options = _gemini_model_options(
        Settings(
            api_providers_enabled=False,
            gemini_api_key="test-key",
            gemini_model="gemini-3-flash-preview",
        )
    )

    assert options == []


async def test_backend_lists_and_fetches_game(tmp_path: Path) -> None:
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
            await annotate_game(
                session,
                game_id=result.game_id,
                persona="technician",
                llm_service=StubLLMService(),
                model="stub-model",
            )

    client = TestClient(create_app(Settings(database_url=db_url)))
    list_response = client.get("/games")
    detail_response = client.get(f"/games/{result.game_id}")

    assert list_response.status_code == 200
    assert list_response.json()[0]["id"] == result.game_id
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["moves"][0]["accepted_uci"] == "e2e4"
    assert detail["moves"][0]["attempts"][0]["legal_ok"] is True
    assert detail["moves"][0]["attempts"][0]["token_usage"]["total_tokens"] > 0
    assert detail["moves"][0]["annotations"][0]["persona"] == "technician"

    report_response = client.get(f"/games/{result.game_id}/report")
    assert report_response.status_code == 200
    assert "# Match Report" in report_response.text


async def test_backend_human_game_accepts_human_move_and_replies(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    client = TestClient(create_app(Settings(database_url=db_url, max_retries=0)))

    start_response = client.post(
        "/human-games/start",
        json={
            "human_color": "white",
            "opponent": "random",
            "max_plies": 2,
        },
    )
    assert start_response.status_code == 200
    state = start_response.json()
    assert state["status"] == "running"
    assert state["turn"] == "white"
    assert "e2e4" in state["legal_moves"]

    move_response = client.post(f"/human-games/{state['id']}/move", json={"move": "e2e4"})
    assert move_response.status_code == 200
    moved_state = move_response.json()
    assert moved_state["status"] == "completed"
    assert moved_state["game_id"] == state["game_id"]
    assert moved_state["termination_reason"] == "max_plies"

    detail_response = client.get(f"/games/{state['game_id']}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["white_player"] == "human"
    assert detail["black_player"] == "random"
    assert [move["move_source"] for move in detail["moves"]] == ["human", "random"]
    assert detail["moves"][0]["accepted_uci"] == "e2e4"


async def test_backend_lists_runs_and_run_games(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name="api-run",
                    competitor_a="random",
                    competitor_b="random",
                    max_plies=1,
                ),
                settings=Settings(max_retries=0),
                source_factory=lambda _name, _rng: RandomMoveSource(),
            )
            await rebuild_game_summaries(session, run_id=result.run_id)

    client = TestClient(create_app(Settings(database_url=db_url)))
    leaderboard_response = client.get(f"/leaderboard?run_id={result.run_id}")
    white_leaderboard_response = client.get(f"/leaderboard?run_id={result.run_id}&color=white")
    events_response = client.get(f"/runs/{result.run_id}/events")
    comparison_response = client.get("/runs/compare")

    assert leaderboard_response.status_code == 200
    assert len(leaderboard_response.json()) == 4
    assert leaderboard_response.json()[0]["games_played"] >= 1
    assert leaderboard_response.json()[0]["avg_game_plies"] == 1.0
    assert "win_rate_ci_low" in leaderboard_response.json()[0]
    assert "illegal_rate_ci_high" in leaderboard_response.json()[0]
    assert white_leaderboard_response.status_code == 200
    assert {row["color"] for row in white_leaderboard_response.json()} == {"white"}
    assert events_response.status_code == 200
    assert isinstance(events_response.json(), list)
    assert comparison_response.status_code == 200
    assert comparison_response.json()[0]["run_id"] == result.run_id
    assert comparison_response.json()[0]["avg_game_plies"] == 1.0
    assert "accuracy_rate" in comparison_response.json()[0]


async def test_backend_compares_models_across_runs(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path}/arena.db"
    await init_db(db_url)
    session_factory = create_session_factory(db_url)

    for name in ("run-a", "run-b"):
        async with session_factory() as session:
            async with session.begin():
                result = await run_tournament(
                    session=session,
                    config=TournamentConfig(
                        name=name,
                        competitor_a="random",
                        competitor_b="random",
                        max_plies=1,
                    ),
                    settings=Settings(max_retries=0),
                    source_factory=lambda _name, _rng: RandomMoveSource(),
                )
                await rebuild_game_summaries(session, run_id=result.run_id)

    client = TestClient(create_app(Settings(database_url=db_url)))
    response = client.get("/models/compare?legality_mode=open&color=all")

    assert response.status_code == 200
    rows = response.json()
    # Both runs pit "random" vs "random" -> one aggregated model row in open mode.
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "random"
    assert row["legality_mode"] == "open"
    assert row["color"] == "all"
    assert row["run_count"] == 2
    assert row["games_played"] == 80
    assert "win_rate_ci_low" in row
    assert "illegal_rate_ci_high" in row

    # Legality filtering keeps modes separate: constrained has no random games here.
    constrained = client.get("/models/compare?legality_mode=constrained")
    assert constrained.status_code == 200
    assert constrained.json() == []


async def test_backend_builds_game_stream_snapshots(tmp_path: Path) -> None:
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

    payload = await _game_stream_payload(session_factory)

    assert payload["games"]
    assert payload["games"][0]["id"] == result.game_id


def test_cpu_offload_uses_configured_mixed_ollama_setting() -> None:
    settings = _settings_for_ollama_options(
        Settings(ollama_cpu_offload_gpu_layers=18),
        sampling=GameDefaults(),
        thinking=False,
        cpu_offload=True,
    )

    assert settings.ollama_num_gpu == 18
    assert settings.ollama_num_ctx == 8192
    assert settings.ollama_num_predict == 256


def test_game_defaults_sampling_flows_into_settings() -> None:
    settings = _settings_for_ollama_options(
        Settings(),
        sampling=GameDefaults(temperature=0.7, top_p=0.9, num_ctx=16384, num_predict=200),
        thinking=False,
        cpu_offload=False,
    )

    assert settings.ollama_temperature == 0.7
    assert settings.ollama_top_p == 0.9
    assert settings.ollama_num_ctx == 16384
    assert settings.ollama_num_predict == 200
    assert settings.ollama_num_gpu is None


def test_stockfish_level_preset_sets_limited_strength() -> None:
    settings = _settings_for_stockfish_level(
        Settings(),
        level="club",
        stockfish_path="/usr/bin/stockfish",
    )

    assert settings.stockfish_path == "/usr/bin/stockfish"
    assert settings.stockfish_skill == 8
    assert settings.stockfish_limit_strength is True
    assert settings.stockfish_target_elo == 1600

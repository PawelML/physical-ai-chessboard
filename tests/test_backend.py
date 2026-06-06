from pathlib import Path

from fastapi.testclient import TestClient

from arena_core.annotations import annotate_game
from arena_core.config import Settings
from arena_core.engine import ArenaGame, RandomMoveSource, StaticMoveSource
from arena_core.leaderboards import rebuild_game_summaries
from arena_core.persistence.database import create_session_factory, init_db
from arena_core.tournaments import TournamentConfig, run_tournament
from backend.main import _game_stream_payload, create_app


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
            await annotate_game(session, game_id=result.game_id, persona="technician")

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
                source_factory=lambda _name: RandomMoveSource(),
            )
            await rebuild_game_summaries(session, run_id=result.run_id)

    client = TestClient(create_app(Settings(database_url=db_url)))
    runs_response = client.get("/runs")
    games_response = client.get(f"/runs/{result.run_id}/games")
    leaderboard_response = client.get(f"/leaderboard?run_id={result.run_id}")
    white_leaderboard_response = client.get(f"/leaderboard?run_id={result.run_id}&color=white")
    events_response = client.get(f"/runs/{result.run_id}/events")
    comparison_response = client.get("/runs/compare")

    assert runs_response.status_code == 200
    assert runs_response.json()[0]["id"] == result.run_id
    assert games_response.status_code == 200
    assert len(games_response.json()) == 4
    assert {game["run_id"] for game in games_response.json()} == {result.run_id}
    assert leaderboard_response.status_code == 200
    assert len(leaderboard_response.json()) == 4
    assert leaderboard_response.json()[0]["games_played"] >= 1
    assert white_leaderboard_response.status_code == 200
    assert {row["color"] for row in white_leaderboard_response.json()} == {"white"}
    assert events_response.status_code == 200
    assert isinstance(events_response.json(), list)
    assert comparison_response.status_code == 200
    assert comparison_response.json()[0]["run_id"] == result.run_id


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

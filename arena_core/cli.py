import asyncio
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, cast

import typer
from sqlalchemy import select

from arena_core.annotations import annotate_game
from arena_core.config import Settings, get_settings
from arena_core.engine import ArenaGame, GameResult, LLMMoveSource, MoveSource, RandomMoveSource
from arena_core.evaluators.stockfish import StockfishEvaluator, StockfishMoveSource
from arena_core.leaderboards import rebuild_game_summaries
from arena_core.llm.providers import llm_service_for
from arena_core.persistence.database import create_session_factory
from arena_core.persistence.database import init_db as create_tables
from arena_core.persistence.models import Attempt
from arena_core.prompts import LegalityMode
from arena_core.reports import export_game_report
from arena_core.tournaments import TournamentConfig, TournamentResult, run_tournament

app = typer.Typer(no_args_is_help=True)


@app.command("init-db")
def init_db(
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
) -> None:
    settings = get_settings()
    asyncio.run(create_tables(db_url or settings.database_url))
    typer.echo(f"Initialized database at {db_url or settings.database_url}")


@app.command("play")
def play(
    white: Annotated[str, typer.Argument(help="White source: random or an Ollama model name.")],
    black: Annotated[str, typer.Argument(help="Black source: random or an Ollama model name.")],
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
    legality_mode: Annotated[
        str,
        typer.Option("--legality-mode", help="open or constrained"),
    ] = "open",
    max_plies: Annotated[
        int | None,
        typer.Option("--max-plies", help="Optional debugging cap."),
    ] = None,
    stockfish_path: Annotated[
        str | None,
        typer.Option("--stockfish-path", help="Enable per-move Stockfish evaluation."),
    ] = None,
) -> None:
    if legality_mode not in {"open", "constrained"}:
        raise typer.BadParameter("legality-mode must be open or constrained")
    settings = get_settings()
    effective_db_url = db_url or settings.database_url
    result, attempt_log = asyncio.run(
        _play_async(
            settings=settings,
            db_url=effective_db_url,
            white_name=white,
            black_name=black,
            legality_mode=legality_mode,
            max_plies=max_plies,
            stockfish_path=stockfish_path,
        )
    )
    typer.echo(f"Game {result.game_id}: {result.result} ({result.termination_reason})")
    typer.echo(result.pgn)
    typer.echo("\nAttempt log:")
    for line in attempt_log:
        typer.echo(line)


@app.command("tournament")
def tournament(
    competitor_a: Annotated[str, typer.Argument(help="First source: random or Ollama model name.")],
    competitor_b: Annotated[
        str,
        typer.Argument(help="Second source: random or Ollama model name."),
    ],
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
    name: Annotated[str, typer.Option("--name", help="Benchmark run name.")] = "starter-run",
    legality_mode: Annotated[
        str,
        typer.Option("--legality-mode", help="open or constrained"),
    ] = "open",
    max_plies: Annotated[
        int | None,
        typer.Option("--max-plies", help="Optional debugging cap per game."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic run seed.")] = 0,
    stockfish_path: Annotated[
        str | None,
        typer.Option("--stockfish-path", help="Enable Stockfish source/evaluation."),
    ] = None,
    stockfish_skill: Annotated[
        int | None,
        typer.Option("--stockfish-skill", help="Optional Stockfish skill level 0-20."),
    ] = None,
    stockfish_elo: Annotated[
        int | None,
        typer.Option("--stockfish-elo", help="Optional UCI_Elo target."),
    ] = None,
) -> None:
    if legality_mode not in {"open", "constrained"}:
        raise typer.BadParameter("legality-mode must be open or constrained")
    settings = get_settings()
    result = asyncio.run(
        _tournament_async(
            settings=settings,
            db_url=db_url or settings.database_url,
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            name=name,
            legality_mode=legality_mode,
            max_plies=max_plies,
            seed=seed,
            stockfish_path=stockfish_path,
            stockfish_skill=stockfish_skill,
            stockfish_elo=stockfish_elo,
        )
    )
    typer.echo(f"Run {result.run_id}: {len(result.game_ids)} games")
    typer.echo(f"Config hash: {result.config_hash}")
    typer.echo("Games: " + ", ".join(str(game_id) for game_id in result.game_ids))


@app.command("rebuild-summaries")
def rebuild_summaries(
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
    run_id: Annotated[int | None, typer.Option("--run-id", help="Optional run id.")] = None,
) -> None:
    settings = get_settings()
    count = asyncio.run(_rebuild_summaries_async(db_url or settings.database_url, run_id))
    typer.echo(f"Rebuilt {count} game summary rows")


@app.command("annotate-game")
def annotate_game_command(
    game_id: Annotated[int, typer.Argument(help="Persisted game id to annotate.")],
    persona: Annotated[str, typer.Option("--persona", help="Persona name.")] = "technician",
    model: Annotated[str, typer.Option("--model", help="Annotation model name.")] = "qwen3.5:9b",
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
) -> None:
    settings = get_settings()
    count = asyncio.run(
        _annotate_game_async(db_url or settings.database_url, game_id, persona, model, settings)
    )
    typer.echo(f"Annotated {count} moves for game {game_id} with persona {persona}")


@app.command("export-report")
def export_report(
    game_id: Annotated[int, typer.Argument(help="Persisted game id to export.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Optional Markdown output path."),
    ] = None,
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
) -> None:
    settings = get_settings()
    report = asyncio.run(_export_report_async(db_url or settings.database_url, game_id))
    if output is None:
        typer.echo(report)
    else:
        output.write_text(report, encoding="utf-8")
        typer.echo(str(output))


async def _play_async(
    *,
    settings: Settings,
    db_url: str,
    white_name: str,
    black_name: str,
    legality_mode: str,
    max_plies: int | None,
    stockfish_path: str | None,
    strategic_memory: bool = False,
    commit_after_each_ply: bool = False,
    on_game_started: Callable[[int], Awaitable[None] | None] | None = None,
) -> tuple[GameResult, list[str]]:
    await create_tables(db_url)
    session_factory = create_session_factory(db_url)
    white = _source_from_name(white_name, settings)
    black = _source_from_name(black_name, settings)
    effective_stockfish_path = stockfish_path or settings.stockfish_path
    evaluator = (
        StockfishEvaluator(
            binary_path=effective_stockfish_path,
            nodes=settings.stockfish_nodes,
            threads=settings.stockfish_threads,
            hash_mb=settings.stockfish_hash_mb,
        )
        if effective_stockfish_path
        else None
    )
    try:
        async with session_factory() as session:
            if commit_after_each_ply:
                game = ArenaGame(
                    white=white,
                    black=black,
                    settings=settings,
                    legality_mode=cast("LegalityMode", legality_mode),
                    max_plies=max_plies,
                    evaluator=evaluator,
                    strategic_memory=strategic_memory,
                )
                result = await game.run(
                    session,
                    commit_after_each_ply=True,
                    on_game_started=on_game_started,
                )
            else:
                async with session.begin():
                    game = ArenaGame(
                        white=white,
                        black=black,
                        settings=settings,
                        legality_mode=cast("LegalityMode", legality_mode),
                        max_plies=max_plies,
                        evaluator=evaluator,
                        strategic_memory=strategic_memory,
                    )
                    result = await game.run(session)

            rows = (
                await session.execute(
                    select(Attempt)
                    .where(Attempt.game_id == result.game_id)
                    .order_by(Attempt.ply, Attempt.attempt_number)
                )
            ).scalars()
            attempt_log = [
                (
                    f"ply={row.ply} attempt={row.attempt_number} "
                    f"move={row.parsed_move or '-'} parse={row.parse_ok} "
                    f"legal={row.legal_ok} error={row.error_type or '-'} "
                    f"think={row.thinking_used} latency_ms={row.latency_ms:.1f}"
                )
                for row in rows
            ]
            return result, attempt_log
    finally:
        if evaluator is not None:
            evaluator.close()


async def _tournament_async(
    *,
    settings: Settings,
    db_url: str,
    competitor_a: str,
    competitor_b: str,
    name: str,
    legality_mode: str,
    max_plies: int | None,
    seed: int,
    stockfish_path: str | None,
    stockfish_skill: int | None,
    stockfish_elo: int | None,
) -> TournamentResult:
    await create_tables(db_url)
    session_factory = create_session_factory(db_url)
    effective_stockfish_path = stockfish_path or settings.stockfish_path
    effective_settings = settings.model_copy(
        update={
            "stockfish_path": effective_stockfish_path,
            "stockfish_skill": stockfish_skill
            if stockfish_skill is not None
            else settings.stockfish_skill,
            "stockfish_target_elo": stockfish_elo
            if stockfish_elo is not None
            else settings.stockfish_target_elo,
            "stockfish_limit_strength": True
            if stockfish_elo is not None
            else settings.stockfish_limit_strength,
        }
    )
    evaluator = (
        StockfishEvaluator(
            binary_path=effective_stockfish_path,
            nodes=effective_settings.stockfish_nodes,
            threads=effective_settings.stockfish_threads,
            hash_mb=effective_settings.stockfish_hash_mb,
        )
        if effective_stockfish_path
        else None
    )
    async with session_factory() as session:
        async with session.begin():
            result = await run_tournament(
                session=session,
                config=TournamentConfig(
                    name=name,
                    competitor_a=competitor_a,
                    competitor_b=competitor_b,
                    legality_mode=cast("LegalityMode", legality_mode),
                    max_plies=max_plies,
                    seed=seed,
                ),
                settings=effective_settings,
                source_factory=lambda source_name, rng: _source_from_name(
                    source_name,
                    effective_settings,
                    rng=rng,
                ),
                evaluator=evaluator,
            )
            await rebuild_game_summaries(session, run_id=result.run_id)
            return result


async def _rebuild_summaries_async(db_url: str, run_id: int | None) -> int:
    session_factory = create_session_factory(db_url)
    async with session_factory() as session:
        async with session.begin():
            return await rebuild_game_summaries(session, run_id=run_id)


async def _annotate_game_async(
    db_url: str,
    game_id: int,
    persona: str,
    model_name: str,
    settings: Settings,
) -> int:
    session_factory = create_session_factory(db_url)
    model, service = llm_service_for(model_name, settings)
    async with session_factory() as session:
        async with session.begin():
            return await annotate_game(
                session,
                game_id=game_id,
                persona=persona,
                llm_service=service,
                model=model,
            )


async def _export_report_async(db_url: str, game_id: int) -> str:
    session_factory = create_session_factory(db_url)
    async with session_factory() as session:
        return await export_game_report(session, game_id=game_id)


def _source_from_name(
    name: str,
    settings: Settings,
    *,
    rng: random.Random | None = None,
) -> MoveSource:
    if name == "random":
        return RandomMoveSource(rng=rng)
    if name == "stockfish":
        if settings.stockfish_path is None:
            raise ValueError("stockfish source requires --stockfish-path or ARENA_STOCKFISH_PATH")
        return StockfishMoveSource(
            binary_path=settings.stockfish_path,
            nodes=settings.stockfish_nodes,
            threads=settings.stockfish_threads,
            hash_mb=settings.stockfish_hash_mb,
            skill=settings.stockfish_skill,
            uci_limit_strength=settings.stockfish_limit_strength,
            target_elo=settings.stockfish_target_elo,
        )
    model, service = llm_service_for(name, settings)
    return LLMMoveSource(model=model, service=service)


@app.command("show-db")
def show_db_path(
    db_url: Annotated[str | None, typer.Option("--db-url", help="SQLAlchemy async DB URL.")] = None,
) -> None:
    value = db_url or get_settings().database_url
    if value.startswith("sqlite+aiosqlite:///"):
        typer.echo(str(Path(value.removeprefix("sqlite+aiosqlite:///")).resolve()))
    else:
        typer.echo(value)

import json
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256

import chess
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from arena_core.config import Settings
from arena_core.engine import ArenaGame, MoveEvaluator, MoveSource
from arena_core.llm.ollama import OllamaModelMetadata, fetch_ollama_model_metadata
from arena_core.llm.providers import parse_provider_model
from arena_core.persistence import models
from arena_core.persistence.repositories import ensure_prompt
from arena_core.prompts import LegalityMode, build_strict_prompt
from arena_core.telemetry import estimate_pair_footprint

SourceFactory = Callable[[str], MoveSource]


@dataclass(frozen=True)
class OpeningSpec:
    eco: str | None
    name: str
    move_sequence: str
    pairing_id: str


@dataclass(frozen=True)
class TournamentConfig:
    name: str
    competitor_a: str
    competitor_b: str
    legality_mode: LegalityMode = "open"
    max_plies: int | None = None
    seed: int = 0
    suite_name: str = "starter"
    suite_version: str = "v1"


@dataclass(frozen=True)
class TournamentResult:
    run_id: int
    config_hash: str
    game_ids: list[int]


STARTER_OPENINGS = [
    OpeningSpec("C20", "King Pawn", "e2e4 e7e5", "starter-king-pawn"),
    OpeningSpec("D00", "Queen Pawn", "d2d4 d7d5", "starter-queen-pawn"),
]


async def run_tournament(
    *,
    session: AsyncSession,
    config: TournamentConfig,
    settings: Settings,
    source_factory: SourceFactory,
    evaluator: MoveEvaluator | None = None,
) -> TournamentResult:
    random.seed(config.seed)
    suite, opening_lines = await ensure_opening_suite(
        session,
        name=config.suite_name,
        version=config.suite_version,
        openings=STARTER_OPENINGS,
    )
    prompt = build_strict_prompt(
        board=chess.Board(),
        san_history=[],
        own_moves=[],
        last_opponent_move=None,
        legality_mode=config.legality_mode,
        version=settings.prompt_version,
    )
    prompt_row = await ensure_prompt(session, prompt)
    model_metadata = await _model_metadata_for_sources(
        [config.competitor_a, config.competitor_b],
        settings=settings,
    )
    config_payload = _config_payload(
        config,
        settings,
        suite.id,
        prompt_row.id,
        model_metadata,
    )
    config_hash = _config_hash(config_payload)
    run = models.BenchmarkRun(
        name=config.name,
        config_hash=config_hash,
        git_commit=_git_commit(),
        seed=config.seed,
        stockfish_version=None,
        stockfish_options=_stockfish_options(settings),
        prompt_id=prompt_row.id,
        opening_suite_id=suite.id,
    )
    session.add(run)
    await session.flush()
    participant_a = await _create_participant(
        session,
        run_id=run.id,
        source_name=config.competitor_a,
        color_policy="both",
        settings=settings,
        model_metadata=model_metadata.get(config.competitor_a),
    )
    participant_b = await _create_participant(
        session,
        run_id=run.id,
        source_name=config.competitor_b,
        color_policy="both",
        settings=settings,
        model_metadata=model_metadata.get(config.competitor_b),
    )
    await _record_pair_telemetry(
        session,
        run_id=run.id,
        competitor_a=config.competitor_a,
        competitor_b=config.competitor_b,
        settings=settings,
    )

    game_ids: list[int] = []
    for line in opening_lines:
        for white_name, black_name, white_participant, black_participant in (
            (config.competitor_a, config.competitor_b, participant_a, participant_b),
            (config.competitor_b, config.competitor_a, participant_b, participant_a),
        ):
            board = board_from_move_sequence(line.move_sequence or "")
            result = await ArenaGame(
                white=source_factory(white_name),
                black=source_factory(black_name),
                settings=settings,
                legality_mode=config.legality_mode,
                max_plies=config.max_plies,
                evaluator=evaluator,
                initial_board=board,
                run_id=run.id,
                white_participant_id=white_participant.id,
                black_participant_id=black_participant.id,
                opening_line_id=line.id,
            ).run(session)
            game_ids.append(result.game_id)
    return TournamentResult(run_id=run.id, config_hash=config_hash, game_ids=game_ids)


async def _record_pair_telemetry(
    session: AsyncSession,
    *,
    run_id: int,
    competitor_a: str,
    competitor_b: str,
    settings: Settings,
) -> None:
    if competitor_a in {"random", "stockfish"} or competitor_b in {"random", "stockfish"}:
        return
    footprint = estimate_pair_footprint(
        competitor_a,
        competitor_b,
        budget_gb=settings.ollama_vram_budget_gb,
    )
    payload = {
        "models": list(footprint.models),
        "footprints": [
            {"model": item.model, "vram_gb": item.vram_gb} for item in footprint.footprints
        ],
        "total_vram_gb": footprint.total_vram_gb,
        "budget_gb": footprint.budget_gb,
        "unknown_models": list(footprint.unknown_models),
    }
    if footprint.exceeds_budget:
        session.add(
            models.OperationalEvent(
                run_id=run_id,
                event_kind="model_swap_warning",
                severity="warning",
                message="Estimated model pair footprint exceeds configured VRAM budget.",
                payload=payload,
            )
        )
    elif footprint.unknown_models:
        session.add(
            models.OperationalEvent(
                run_id=run_id,
                event_kind="model_footprint_unknown",
                severity="info",
                message="One or more model footprints are unknown; swap risk cannot be estimated.",
                payload=payload,
            )
        )
    else:
        session.add(
            models.OperationalEvent(
                run_id=run_id,
                event_kind="model_pair_fit",
                severity="info",
                message="Estimated model pair footprint fits within configured VRAM budget.",
                payload=payload,
            )
        )


async def ensure_opening_suite(
    session: AsyncSession,
    *,
    name: str,
    version: str,
    openings: list[OpeningSpec],
) -> tuple[models.OpeningSuite, list[models.OpeningLine]]:
    suite = (
        await session.execute(
            select(models.OpeningSuite).where(
                models.OpeningSuite.name == name,
                models.OpeningSuite.version == version,
            )
        )
    ).scalar_one_or_none()
    if suite is None:
        suite = models.OpeningSuite(name=name, version=version, notes="Built-in starter suite")
        session.add(suite)
        await session.flush()

    lines: list[models.OpeningLine] = []
    for index, opening in enumerate(openings, start=1):
        line = (
            await session.execute(
                select(models.OpeningLine).where(
                    models.OpeningLine.suite_id == suite.id,
                    models.OpeningLine.name == opening.name,
                )
            )
        ).scalar_one_or_none()
        if line is None:
            ply_start = len(opening.move_sequence.split())
            line = models.OpeningLine(
                suite_id=suite.id,
                eco=opening.eco,
                name=opening.name,
                move_sequence=opening.move_sequence,
                intended_ply_start=ply_start,
                pairing_id=opening.pairing_id or f"{name}-{version}-{index}",
            )
            session.add(line)
            await session.flush()
        lines.append(line)
    return suite, lines


def board_from_move_sequence(move_sequence: str) -> chess.Board:
    board = chess.Board()
    for uci in move_sequence.split():
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Opening move {uci} is illegal after {board.fen()}")
        board.push(move)
    return board


async def _create_participant(
    session: AsyncSession,
    *,
    run_id: int,
    source_name: str,
    color_policy: str,
    settings: Settings,
    model_metadata: OllamaModelMetadata | None,
) -> models.RunParticipant:
    model_snapshot_id = None
    if source_name == "random":
        opponent_type = "random"
    elif source_name == "stockfish":
        opponent_type = "stockfish"
    else:
        opponent_type = "model"
    if opponent_type == "model":
        provider_model = parse_provider_model(source_name)
        model = models.Model(
            provider=provider_model.provider,
            name=provider_model.model,
            family=model_metadata.family if model_metadata else None,
            param_size=model_metadata.parameter_size if model_metadata else None,
            modality="text",
            is_local=provider_model.provider == "local",
        )
        session.add(model)
        await session.flush()
        snapshot = models.ModelSnapshot(
            model_id=model.id,
            ollama_digest=model_metadata.digest if model_metadata else None,
            quantization=model_metadata.quantization if model_metadata else None,
            context_window=model_metadata.context_window if model_metadata else None,
            sampler_params=_ollama_sampler_params(),
            runtime_version=model_metadata.runtime_version if model_metadata else None,
        )
        session.add(snapshot)
        await session.flush()
        model_snapshot_id = snapshot.id
    participant = models.RunParticipant(
        run_id=run_id,
        model_snapshot_id=model_snapshot_id,
        opponent_type=opponent_type,
        stockfish_skill=settings.stockfish_skill if opponent_type == "stockfish" else None,
        uci_limit_strength=settings.stockfish_limit_strength
        if opponent_type == "stockfish"
        else None,
        target_elo=settings.stockfish_target_elo if opponent_type == "stockfish" else None,
        color_policy=color_policy,
        display_name=source_name,
    )
    session.add(participant)
    await session.flush()
    return participant


def _config_payload(
    config: TournamentConfig,
    settings: Settings,
    opening_suite_id: int,
    prompt_id: int,
    model_metadata: dict[str, OllamaModelMetadata],
) -> dict[str, object]:
    return {
        "name": config.name,
        "competitors": [config.competitor_a, config.competitor_b],
        "model_snapshots": {
            source_name: _model_metadata_payload(metadata)
            for source_name, metadata in model_metadata.items()
        },
        "legality_mode": config.legality_mode,
        "max_plies": config.max_plies,
        "seed": config.seed,
        "opening_suite_id": opening_suite_id,
        "prompt_id": prompt_id,
        "prompt_version": settings.prompt_version,
        "stockfish_options": _stockfish_options(settings),
    }


async def _model_metadata_for_sources(
    source_names: list[str],
    *,
    settings: Settings,
) -> dict[str, OllamaModelMetadata]:
    metadata: dict[str, OllamaModelMetadata] = {}
    for source_name in dict.fromkeys(source_names):
        provider_model = parse_provider_model(source_name)
        if provider_model.provider != "local" or provider_model.model in {"random", "stockfish"}:
            continue
        item = await fetch_ollama_model_metadata(
            model=provider_model.model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
        if item is not None:
            metadata[source_name] = item
    return metadata


def _model_metadata_payload(metadata: OllamaModelMetadata) -> dict[str, object]:
    return {
        "name": metadata.name,
        "digest": metadata.digest,
        "family": metadata.family,
        "parameter_size": metadata.parameter_size,
        "quantization": metadata.quantization,
        "context_window": metadata.context_window,
        "runtime_version": metadata.runtime_version,
        "modified_at": metadata.modified_at,
        "size_bytes": metadata.size_bytes,
        "sampler_params": _ollama_sampler_params(),
    }


def _ollama_sampler_params() -> dict[str, object]:
    return {"temperature": 0, "format": "json", "think": False}


def _config_hash(payload: dict[str, object]) -> str:
    return sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _stockfish_options(settings: Settings) -> dict[str, object] | None:
    if settings.stockfish_path is None:
        return None
    return {
        "path": settings.stockfish_path,
        "nodes": settings.stockfish_nodes,
        "threads": settings.stockfish_threads,
        "hash_mb": settings.stockfish_hash_mb,
        "skill": settings.stockfish_skill,
        "limit_strength": settings.stockfish_limit_strength,
        "target_elo": settings.stockfish_target_elo,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    if not commit or commit == "HEAD":
        return None
    return commit

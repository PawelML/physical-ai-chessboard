"""Build candidate-level training rows for a learned chess move critic/ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO

import chess

from arena_core.evaluators.stockfish import EngineEvaluation, StockfishEvaluator
from finetune._common import write_metadata_sidecar


class MoveEvaluator(Protocol):
    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        """Evaluate a legal candidate move."""

    def close(self) -> None:
        """Release evaluator resources."""


@dataclass(frozen=True)
class CriticRankerConfig:
    db: str
    output: str
    metadata_output: str | None
    stockfish_path: str
    stockfish_nodes: int = 200_000
    stockfish_hash_mb: int = 128
    stockfish_threads: int = 1
    evaluation_cache: str | None = None
    candidate_inputs: list[str] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    max_positions: int | None = None
    max_candidates_per_position: int = 6
    random_legal_per_position: int = 1
    include_stockfish_best: bool = True
    max_opponent_replies: int = 40
    seed: int = 0
    shuffle_positions: bool = False


@dataclass(frozen=True)
class CandidateSpec:
    fen_before: str
    candidate_uci: str
    source: str
    candidate_rank_in_generator: int | None = None
    generator_idea: str | None = None
    source_run_id: int | None = None
    source_game_id: int | None = None
    source_move_id: int | None = None
    source_attempt_id: int | None = None


@dataclass
class CriticRankerStats:
    positions_seen: int = 0
    positions_written: int = 0
    candidates_seen: int = 0
    rows_written: int = 0
    dropped_duplicate_candidate: int = 0
    dropped_illegal_candidate: int = 0
    dropped_candidate_limit: int = 0
    engine_evaluations: int = 0
    evaluation_cache_hits: int = 0
    evaluation_cache_misses: int = 0
    by_source: Counter[str] = field(default_factory=Counter)
    by_risk: Counter[str] = field(default_factory=Counter)
    by_split: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "positions_seen": self.positions_seen,
            "positions_written": self.positions_written,
            "candidates_seen": self.candidates_seen,
            "rows_written": self.rows_written,
            "dropped_duplicate_candidate": self.dropped_duplicate_candidate,
            "dropped_illegal_candidate": self.dropped_illegal_candidate,
            "dropped_candidate_limit": self.dropped_candidate_limit,
            "engine_evaluations": self.engine_evaluations,
            "evaluation_cache_hits": self.evaluation_cache_hits,
            "evaluation_cache_misses": self.evaluation_cache_misses,
            "by_source": dict(sorted(self.by_source.items())),
            "by_risk": dict(sorted(self.by_risk.items())),
            "by_split": dict(sorted(self.by_split.items())),
        }


def main() -> None:
    args = _parse_args()
    config = CriticRankerConfig(
        db=str(args.db),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        stockfish_path=str(args.stockfish_path),
        stockfish_nodes=args.stockfish_nodes,
        stockfish_hash_mb=args.stockfish_hash_mb,
        stockfish_threads=args.stockfish_threads,
        evaluation_cache=str(args.evaluation_cache) if args.evaluation_cache else None,
        candidate_inputs=[str(path) for path in args.candidate_input],
        run_ids=args.run_id,
        max_positions=args.max_positions,
        max_candidates_per_position=args.max_candidates_per_position,
        random_legal_per_position=args.random_legal_per_position,
        include_stockfish_best=not args.no_stockfish_best,
        max_opponent_replies=args.max_opponent_replies,
        seed=args.seed,
        shuffle_positions=args.shuffle_positions,
    )
    evaluator = StockfishEvaluator(
        binary_path=config.stockfish_path,
        nodes=config.stockfish_nodes,
        threads=config.stockfish_threads,
        hash_mb=config.stockfish_hash_mb,
    )
    cached_evaluator: MoveEvaluator
    if config.evaluation_cache is not None:
        cached_evaluator = CachedMoveEvaluator(
            evaluator,
            cache_path=Path(config.evaluation_cache),
        )
    else:
        cached_evaluator = evaluator
    try:
        stats = build_critic_ranker_dataset(config, evaluator=cached_evaluator)
    finally:
        cached_evaluator.close()
    if config.metadata_output is not None:
        write_metadata_sidecar(Path(config.metadata_output), config=config, stats=stats)
    print(
        f"Wrote {stats.rows_written} critic-ranker rows from "
        f"{stats.positions_written} positions to {config.output}."
    )


def build_critic_ranker_dataset(
    config: CriticRankerConfig,
    *,
    evaluator: MoveEvaluator,
) -> CriticRankerStats:
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = CriticRankerStats()
    rng = random.Random(config.seed)

    con = sqlite3.connect(config.db)
    con.row_factory = sqlite3.Row
    try:
        candidates_by_fen = _candidate_specs_by_fen(con, config=config, rng=rng, stats=stats)
    finally:
        con.close()

    position_items = list(candidates_by_fen.items())
    if config.shuffle_positions:
        rng.shuffle(position_items)

    with output_path.open("w", encoding="utf-8") as output_file:
        for _fen, specs in position_items:
            if config.max_positions is not None and stats.positions_written >= config.max_positions:
                break
            written_for_position = 0
            for spec in specs:
                if written_for_position >= config.max_candidates_per_position:
                    stats.dropped_candidate_limit += 1
                    continue
                row = _build_row(
                    spec=spec,
                    evaluator=evaluator,
                    max_opponent_replies=config.max_opponent_replies,
                )
                if row is None:
                    stats.dropped_illegal_candidate += 1
                    continue
                output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                stats.rows_written += 1
                stats.engine_evaluations += 1
                stats.by_source[spec.source] += 1
                stats.by_risk[str(row["risk"])] += 1
                stats.by_split[str(row["split"])] += 1
                written_for_position += 1
            if written_for_position > 0:
                stats.positions_written += 1
    stats.evaluation_cache_hits = int(getattr(evaluator, "cache_hits", 0))
    stats.evaluation_cache_misses = int(getattr(evaluator, "cache_misses", 0))
    return stats


class CachedMoveEvaluator:
    def __init__(self, inner: MoveEvaluator, *, cache_path: Path) -> None:
        self.inner = inner
        self.cache_path = cache_path
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache = _load_evaluation_cache(cache_path)
        self._handle: TextIO | None = None

    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        fen = board_before.fen()
        uci = move.uci()
        key = _evaluation_cache_key(fen, uci)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        evaluation = self.inner.evaluate_move(board_before, move)
        self._cache[key] = evaluation
        self._append_cache_row(fen=fen, uci=uci, key=key, evaluation=evaluation)
        return evaluation

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.inner.close()

    def _append_cache_row(
        self,
        *,
        fen: str,
        uci: str,
        key: str,
        evaluation: EngineEvaluation,
    ) -> None:
        if self._handle is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.cache_path.open("a", encoding="utf-8")
        self._handle.write(
            json.dumps(
                {
                    "key": key,
                    "fen_before": fen,
                    "candidate_uci": uci,
                    "evaluation": asdict(evaluation),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._handle.flush()


def _load_evaluation_cache(cache_path: Path) -> dict[str, EngineEvaluation]:
    if not cache_path.exists():
        return {}
    cache: dict[str, EngineEvaluation] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                key = str(payload["key"])
                evaluation_payload = payload["evaluation"]
                if isinstance(evaluation_payload, dict):
                    cache[key] = EngineEvaluation(**evaluation_payload)
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return cache


def _evaluation_cache_key(fen: str, uci: str) -> str:
    return hashlib.sha256(f"{fen}|{uci}".encode()).hexdigest()


def _candidate_specs_by_fen(
    con: sqlite3.Connection,
    *,
    config: CriticRankerConfig,
    rng: random.Random,
    stats: CriticRankerStats,
) -> dict[str, list[CandidateSpec]]:
    grouped: dict[str, list[CandidateSpec]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()

    for spec in _accepted_move_specs(con, run_ids=config.run_ids):
        _add_spec(grouped=grouped, seen=seen, spec=spec, stats=stats)
    for spec in _metadata_candidate_specs(con, run_ids=config.run_ids):
        _add_spec(grouped=grouped, seen=seen, spec=spec, stats=stats)
    if config.include_stockfish_best:
        for spec in _stockfish_best_specs(con, run_ids=config.run_ids):
            _add_spec(grouped=grouped, seen=seen, spec=spec, stats=stats)
    for spec in _external_candidate_specs(config.candidate_inputs):
        _add_spec(grouped=grouped, seen=seen, spec=spec, stats=stats)
    if config.random_legal_per_position > 0:
        for spec in _random_legal_specs(
            grouped.keys(),
            count_per_position=config.random_legal_per_position,
            rng=rng,
        ):
            _add_spec(grouped=grouped, seen=seen, spec=spec, stats=stats)

    stats.positions_seen = len(grouped)
    return dict(grouped)


def _add_spec(
    *,
    grouped: dict[str, list[CandidateSpec]],
    seen: set[tuple[str, str]],
    spec: CandidateSpec,
    stats: CriticRankerStats,
) -> None:
    stats.candidates_seen += 1
    key = (spec.fen_before, spec.candidate_uci)
    if key in seen:
        stats.dropped_duplicate_candidate += 1
        return
    board = chess.Board(spec.fen_before)
    try:
        move = chess.Move.from_uci(spec.candidate_uci)
    except ValueError:
        stats.dropped_illegal_candidate += 1
        return
    if move not in board.legal_moves:
        stats.dropped_illegal_candidate += 1
        return
    seen.add(key)
    grouped[spec.fen_before].append(spec)


def _accepted_move_specs(
    con: sqlite3.Connection,
    *,
    run_ids: list[int],
) -> Iterable[CandidateSpec]:
    where, params = _run_filter("g.run_id", run_ids)
    sql = f"""
        select
            g.run_id,
            g.id as game_id,
            m.id as move_id,
            m.fen_before,
            m.accepted_uci,
            e.classification
        from moves m
        join games g on g.id = m.game_id
        left join engine_evaluations e on e.move_id = m.id
        {where}
        order by g.run_id, g.id, m.ply
    """
    for row in con.execute(sql, params):
        classification = str(row["classification"] or "")
        source = "arena_blunder" if classification in {"blunder", "mate_missed"} else "arena_move"
        yield CandidateSpec(
            fen_before=str(row["fen_before"]),
            candidate_uci=str(row["accepted_uci"]),
            source=source,
            source_run_id=int(row["run_id"]),
            source_game_id=int(row["game_id"]),
            source_move_id=int(row["move_id"]),
        )


def _metadata_candidate_specs(
    con: sqlite3.Connection,
    *,
    run_ids: list[int],
) -> Iterable[CandidateSpec]:
    where, params = _run_filter("g.run_id", run_ids)
    sql = f"""
        select
            g.run_id,
            g.id as game_id,
            m.id as move_id,
            m.fen_before,
            a.id as attempt_id,
            a.reranker_metadata
        from attempts a
        join games g on g.id = a.game_id
        left join moves m on m.id = a.move_id
        {where}
          and a.reranker_metadata is not null
          and m.fen_before is not null
        order by g.run_id, g.id, a.ply, a.attempt_number
    """
    for row in con.execute(sql, params):
        metadata = _json_object(row["reranker_metadata"])
        if metadata.get("regime") != "llm_deliberative":
            continue
        candidates = metadata.get("candidate_legal_moves")
        if not isinstance(candidates, list):
            continue
        raw_candidates = _raw_candidates_by_uci(metadata)
        for index, move_text in enumerate(candidates):
            if not isinstance(move_text, str):
                continue
            raw = raw_candidates.get(move_text)
            yield CandidateSpec(
                fen_before=str(row["fen_before"]),
                candidate_uci=move_text,
                source="arena_candidate",
                candidate_rank_in_generator=index,
                generator_idea=raw.get("idea") if raw else None,
                source_run_id=int(row["run_id"]),
                source_game_id=int(row["game_id"]),
                source_move_id=int(row["move_id"]) if row["move_id"] is not None else None,
                source_attempt_id=int(row["attempt_id"]),
            )


def _stockfish_best_specs(
    con: sqlite3.Connection,
    *,
    run_ids: list[int],
) -> Iterable[CandidateSpec]:
    where, params = _run_filter("g.run_id", run_ids)
    sql = f"""
        select distinct
            g.run_id,
            g.id as game_id,
            m.id as move_id,
            m.fen_before,
            e.best_move_uci
        from moves m
        join games g on g.id = m.game_id
        join engine_evaluations e on e.move_id = m.id
        {where}
          and e.best_move_uci is not null
        order by g.run_id, g.id, m.ply
    """
    for row in con.execute(sql, params):
        yield CandidateSpec(
            fen_before=str(row["fen_before"]),
            candidate_uci=str(row["best_move_uci"]),
            source="stockfish_good",
            source_run_id=int(row["run_id"]),
            source_game_id=int(row["game_id"]),
            source_move_id=int(row["move_id"]),
        )


def _random_legal_specs(
    fens: Iterable[str],
    *,
    count_per_position: int,
    rng: random.Random,
) -> Iterable[CandidateSpec]:
    for fen in fens:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        rng.shuffle(moves)
        for move in moves[:count_per_position]:
            yield CandidateSpec(
                fen_before=fen,
                candidate_uci=move.uci(),
                source="random_legal",
            )


def _external_candidate_specs(paths: list[str]) -> Iterable[CandidateSpec]:
    for path_text in paths:
        path = Path(path_text)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    continue
                fen = payload.get("fen_before") or payload.get("fen")
                move = (
                    payload.get("candidate_uci")
                    or payload.get("uci")
                    or payload.get("move")
                )
                if not isinstance(fen, str) or not isinstance(move, str):
                    continue
                yield CandidateSpec(
                    fen_before=fen,
                    candidate_uci=move,
                    source=str(payload.get("source") or "policy_sample"),
                    candidate_rank_in_generator=_optional_int(
                        payload.get("candidate_rank_in_generator")
                    ),
                    generator_idea=(
                        str(payload["generator_idea"])
                        if isinstance(payload.get("generator_idea"), str)
                        else None
                    ),
                )


def _build_row(
    *,
    spec: CandidateSpec,
    evaluator: MoveEvaluator,
    max_opponent_replies: int,
) -> dict[str, Any] | None:
    board = chess.Board(spec.fen_before)
    move = chess.Move.from_uci(spec.candidate_uci)
    if move not in board.legal_moves:
        return None
    candidate_san = board.san(move)
    board_after = board.copy(stack=False)
    board_after.push(move)
    evaluation = evaluator.evaluate_move(board, move)
    risk = risk_from_evaluation(evaluation)
    score = score_from_evaluation(evaluation)
    reason_type = reason_type_from_evaluation(evaluation)
    target = {
        "risk": risk,
        "blunder": risk == "blunder",
        "score": score,
        "reason_type": reason_type,
    }
    prompt = build_critic_prompt(
        fen_before=spec.fen_before,
        side_to_move=_side_name(board.turn),
        candidate_uci=spec.candidate_uci,
        candidate_san=candidate_san,
        fen_after=board_after.fen(),
        opponent_legal_replies=_legal_moves_text(board_after, max_opponent_replies),
    )
    return {
        "prompt": prompt,
        "completion": json.dumps(target, separators=(",", ":")),
        "fen_before": spec.fen_before,
        "side_to_move": _side_name(board.turn),
        "candidate_uci": spec.candidate_uci,
        "candidate_san": candidate_san,
        "fen_after": board_after.fen(),
        "source": spec.source,
        "split": split_for_fen(spec.fen_before),
        "legal_move_count": board.legal_moves.count(),
        "candidate_rank_in_generator": spec.candidate_rank_in_generator,
        "generator_idea": spec.generator_idea,
        "source_run_id": spec.source_run_id,
        "source_game_id": spec.source_game_id,
        "source_move_id": spec.source_move_id,
        "source_attempt_id": spec.source_attempt_id,
        "stockfish_best_uci": evaluation.best_move_uci,
        "centipawn_loss": evaluation.centipawn_loss,
        "classification": evaluation.classification,
        "risk": risk,
        "score": score,
        "reason_type": reason_type,
        "target": json.dumps(target, separators=(",", ":")),
    }


def build_critic_prompt(
    *,
    fen_before: str,
    side_to_move: str,
    candidate_uci: str,
    candidate_san: str,
    fen_after: str,
    opponent_legal_replies: str,
) -> str:
    return "\n".join(
        [
            "You are a chess move risk classifier.",
            "",
            "Position:",
            f"FEN before: {fen_before}",
            f"Side to move: {side_to_move}",
            f"Candidate move: {candidate_uci} ({candidate_san})",
            f"FEN after candidate: {fen_after}",
            f"Opponent legal replies: {opponent_legal_replies}",
            "",
            "Classify whether the candidate is tactically safe.",
            "Return JSON only:",
            '{"risk":"blunder|mistake|playable|good","blunder":true,'
            '"score":0,"reason_type":"material_or_mate"}',
        ]
    )


def risk_from_evaluation(evaluation: EngineEvaluation) -> str:
    if evaluation.classification in {"blunder", "mate_missed"}:
        return "blunder"
    cpl = evaluation.centipawn_loss
    if cpl is None:
        return "blunder" if evaluation.classification == "mate_missed" else "playable"
    if cpl <= 50:
        return "good"
    if cpl <= 150:
        return "playable"
    if cpl <= 300:
        return "mistake"
    return "blunder"


def score_from_evaluation(evaluation: EngineEvaluation) -> int:
    if evaluation.classification == "mate_missed":
        return 0
    cpl = evaluation.centipawn_loss
    if cpl is None:
        return 50
    return max(min(round(100 - (cpl / 4)), 100), 0)


def reason_type_from_evaluation(evaluation: EngineEvaluation) -> str:
    if evaluation.classification == "mate_missed":
        return "material_or_mate"
    cpl = evaluation.centipawn_loss
    if cpl is not None and cpl > 300:
        return "material_or_mate"
    if risk_from_evaluation(evaluation) == "good":
        return "none"
    return "unknown"


def split_for_fen(fen: str) -> str:
    digest = hashlib.sha256(fen.encode()).digest()
    bucket = digest[0] % 10
    if bucket == 0:
        return "validation"
    if bucket == 1:
        return "test"
    return "train"


def _run_filter(column: str, run_ids: list[int]) -> tuple[str, list[int]]:
    if not run_ids:
        return "where 1 = 1", []
    placeholders = ",".join("?" for _ in run_ids)
    return f"where {column} in ({placeholders})", list(run_ids)


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _raw_candidates_by_uci(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = _json_object(metadata.get("candidate_raw_response"))
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        move = item.get("move")
        if isinstance(move, str):
            result[move] = item
    return result


def _legal_moves_text(board: chess.Board, limit: int) -> str:
    moves = sorted(move.uci() for move in board.legal_moves)
    if limit >= 0 and len(moves) > limit:
        visible = moves[:limit]
        return f"{', '.join(visible)} ... ({len(moves) - limit} more)"
    return ", ".join(moves)


def _side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build candidate-level rows for a learned chess critic/ranker."
    )
    parser.add_argument("--db", type=Path, default=Path("arena.db"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--stockfish-path", type=Path, required=True)
    parser.add_argument("--stockfish-nodes", type=int, default=200_000)
    parser.add_argument("--stockfish-hash-mb", type=int, default=128)
    parser.add_argument("--stockfish-threads", type=int, default=1)
    parser.add_argument("--evaluation-cache", type=Path)
    parser.add_argument("--candidate-input", type=Path, action="append", default=[])
    parser.add_argument("--run-id", type=int, action="append", default=[])
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--max-candidates-per-position", type=int, default=6)
    parser.add_argument("--random-legal-per-position", type=int, default=1)
    parser.add_argument("--no-stockfish-best", action="store_true")
    parser.add_argument("--max-opponent-replies", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-positions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

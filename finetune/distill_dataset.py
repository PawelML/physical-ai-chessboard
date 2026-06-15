"""Relabel an existing fine-tuning JSONL with Stockfish engine labels.

Reads the JSONL produced by ``finetune.build_dataset`` (or
``scripts/build_finetune_dataset.py``), evaluates every position with the
arena's own ``StockfishEvaluator``, and writes a new JSONL with the same
byte-identical prompts but engine-derived labels:

- ``--label-mode distill``: replace the human completion with Stockfish's best
  move (engine distillation, Variant B of the fine-tuning plan).
- ``--label-mode filter``: keep the human completion, but drop examples where
  the human move loses more than ``--max-cpl`` centipawns (or misses a mate).

Prompts never contain the target move, so relabeling does not require
re-rendering; ``prompt_version`` and ``template_hash`` carry over unchanged.
Labeling is CPU-only and parallelized over ``--workers`` engine processes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from multiprocessing import Pool
from pathlib import Path
from typing import TextIO

import chess

from finetune._common import (
    init_stockfish_worker,
    stockfish_worker_evaluator,
    write_metadata_sidecar,
)
from finetune.chess_reward import require_stockfish_path

PROGRESS_EVERY_ROWS = 500


@dataclass(frozen=True)
class DistillConfig:
    input: str
    output: str
    label_mode: str
    nodes: int
    max_cpl: int
    workers: int
    hash_mb: int
    dedup: bool
    limit: int | None
    stockfish_path: str


@dataclass
class DistillStats:
    rows_read: int = 0
    rows_written: int = 0
    dropped_invalid_row: int = 0
    dropped_no_best_move: int = 0
    dropped_filtered: int = 0
    dropped_duplicate: int = 0
    human_matches_engine: int = 0
    human_cpl_sum: int = 0
    human_cpl_count: int = 0
    human_classifications: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, object]:
        labeled = self.human_cpl_count
        return {
            "rows_read": self.rows_read,
            "rows_written": self.rows_written,
            "dropped_invalid_row": self.dropped_invalid_row,
            "dropped_no_best_move": self.dropped_no_best_move,
            "dropped_filtered": self.dropped_filtered,
            "dropped_duplicate": self.dropped_duplicate,
            "human_matches_engine": self.human_matches_engine,
            "human_engine_agreement_rate": (
                round(self.human_matches_engine / self.rows_read, 4) if self.rows_read else None
            ),
            "human_avg_cpl": round(self.human_cpl_sum / labeled, 2) if labeled else None,
            "human_classifications": dict(sorted(self.human_classifications.items())),
        }


def main() -> None:
    args = _parse_args()
    stockfish_path = require_stockfish_path(args.stockfish_path)
    config = DistillConfig(
        input=str(args.input),
        output=str(args.output),
        label_mode=args.label_mode,
        nodes=args.nodes,
        max_cpl=args.max_cpl,
        workers=args.workers,
        hash_mb=args.hash_mb,
        dedup=not args.no_dedup,
        limit=args.limit,
        stockfish_path=stockfish_path,
    )
    stats = relabel_file(config)
    stats_json = stats.to_json()
    if args.metadata_output is not None:
        write_metadata_sidecar(args.metadata_output, config=config, stats=stats)
    print(
        f"Wrote {stats.rows_written}/{stats.rows_read} rows "
        f"({config.label_mode} mode); "
        f"human/engine agreement {stats_json['human_engine_agreement_rate']}, "
        f"human avg CPL {stats_json['human_avg_cpl']}."
    )


def relabel_file(config: DistillConfig) -> DistillStats:
    input_path = Path(config.input)
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = DistillStats()
    seen_examples: set[str] = set()
    started = time.monotonic()

    with input_path.open(encoding="utf-8") as input_file:
        lines = _limited(input_file, config.limit)
        with output_path.open("w", encoding="utf-8") as output_file:
            with Pool(
                processes=config.workers,
                initializer=init_stockfish_worker,
                initargs=(config.stockfish_path, config.nodes, config.hash_mb),
            ) as pool:
                for row, label in pool.imap(_label_row, lines, chunksize=8):
                    stats.rows_read += 1
                    _process_result(
                        row=row,
                        label=label,
                        config=config,
                        stats=stats,
                        seen_examples=seen_examples,
                        output_file=output_file,
                    )
                    if stats.rows_read % PROGRESS_EVERY_ROWS == 0:
                        elapsed = time.monotonic() - started
                        rate = stats.rows_read / elapsed if elapsed > 0 else 0.0
                        print(
                            f"  {stats.rows_read} rows labeled "
                            f"({stats.rows_written} written, {rate:.1f} rows/s)",
                            file=sys.stderr,
                        )
    return stats


def _process_result(
    *,
    row: dict[str, object] | None,
    label: dict[str, object] | None,
    config: DistillConfig,
    stats: DistillStats,
    seen_examples: set[str],
    output_file: TextIO,
) -> None:
    if row is None or label is None:
        stats.dropped_invalid_row += 1
        return

    best_move = label["best_move_uci"]
    human_cpl = label["human_centipawn_loss"]
    classification = str(label["human_classification"])
    stats.human_classifications[classification] += 1
    if best_move == label["human_move_uci"]:
        stats.human_matches_engine += 1
    if isinstance(human_cpl, int):
        stats.human_cpl_sum += human_cpl
        stats.human_cpl_count += 1

    if config.label_mode == "distill":
        if best_move is None:
            stats.dropped_no_best_move += 1
            return
        row["completion"] = f'{{"move":"{best_move}"}}'
        row["move"] = best_move
        row["san"] = label["best_move_san"]
    else:  # filter
        missed_mate = classification == "mate_missed"
        too_lossy = isinstance(human_cpl, int) and human_cpl > config.max_cpl
        if missed_mate or too_lossy:
            stats.dropped_filtered += 1
            return
    row["engine_label"] = label

    if config.dedup:
        key = sha256(
            f"{row['prompt']}\x00{row['completion']}".encode()
        ).hexdigest()
        if key in seen_examples:
            stats.dropped_duplicate += 1
            return
        seen_examples.add(key)

    output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    stats.rows_written += 1


def _label_row(line: str) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    evaluator = stockfish_worker_evaluator()
    try:
        row = json.loads(line)
        board = chess.Board(str(row["fen"]))
        human_move = chess.Move.from_uci(str(row["move"]))
        if human_move not in board.legal_moves:
            return None, None
    except (ValueError, KeyError, TypeError):
        return None, None

    evaluation = evaluator.evaluate_move(board, human_move)
    best_move_san: str | None = None
    if evaluation.best_move_uci is not None:
        try:
            best_move_san = board.san(chess.Move.from_uci(evaluation.best_move_uci))
        except ValueError:
            return None, None
    label: dict[str, object] = {
        "engine_version": evaluation.engine_version,
        "nodes": evaluation.nodes,
        "best_move_uci": evaluation.best_move_uci,
        "best_move_san": best_move_san,
        "human_move_uci": human_move.uci(),
        "human_san": row.get("san"),
        "human_centipawn_loss": evaluation.centipawn_loss,
        "human_classification": evaluation.classification,
        "eval_before_cp": evaluation.eval_before_cp,
        "mate_before": evaluation.mate_before,
    }
    return row, label


def _limited(input_file: TextIO, limit: int | None) -> Iterator[str]:
    for index, line in enumerate(input_file):
        if limit is not None and index >= limit:
            return
        if line.strip():
            yield line


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel a fine-tuning JSONL with Stockfish labels: replace human moves "
            "with engine best moves (distill) or drop human blunders (filter)."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional sidecar JSON metadata path (config + label stats).",
    )
    parser.add_argument(
        "--label-mode",
        choices=("distill", "filter"),
        default="distill",
        help="distill: target = engine best move; filter: keep human move, drop blunders.",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=100_000,
        help="Fixed engine nodes per analysis (teacher strength; two analyses per row).",
    )
    parser.add_argument(
        "--max-cpl",
        type=int,
        default=100,
        help="filter mode: drop rows where the human move loses more centipawns than this.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help="Parallel single-threaded engine processes.",
    )
    parser.add_argument("--hash-mb", type=int, default=64, help="Engine hash per worker.")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Keep exact duplicate (prompt, completion) rows instead of dropping them.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N rows (smoke runs).")
    parser.add_argument(
        "--stockfish-path",
        help="Stockfish binary; defaults to ARENA_STOCKFISH_PATH.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

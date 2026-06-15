"""Build a GRPO dataset slice from arena positions where models made tactical errors."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finetune._common import write_metadata_sidecar


@dataclass(frozen=True)
class ErrorMineConfig:
    db: str
    output: str
    metadata_output: str | None
    run_ids: list[int]
    min_cpl: int
    include_blunders: bool
    include_mate_missed: bool
    include_high_mistakes: bool
    max_examples: int | None


@dataclass
class ErrorMineStats:
    candidates: int = 0
    written: int = 0
    dropped_duplicate_fen: int = 0
    dropped_missing_prompt: int = 0
    dropped_best_equals_accepted: int = 0
    by_run: Counter[int] = field(default_factory=Counter)
    by_classification: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "written": self.written,
            "dropped_duplicate_fen": self.dropped_duplicate_fen,
            "dropped_missing_prompt": self.dropped_missing_prompt,
            "dropped_best_equals_accepted": self.dropped_best_equals_accepted,
            "by_run": dict(sorted(self.by_run.items())),
            "by_classification": dict(sorted(self.by_classification.items())),
        }


def main() -> None:
    args = _parse_args()
    config = ErrorMineConfig(
        db=str(args.db),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        run_ids=args.run_id,
        min_cpl=args.min_cpl,
        include_blunders=not args.no_blunders,
        include_mate_missed=not args.no_mate_missed,
        include_high_mistakes=not args.no_high_mistakes,
        max_examples=args.max_examples,
    )
    stats = build_error_mined_dataset(config)
    if config.metadata_output is not None:
        write_metadata_sidecar(Path(config.metadata_output), config=config, stats=stats)
    print(
        f"Wrote {stats.written}/{stats.candidates} error-mined rows to {config.output}; "
        f"dropped {stats.dropped_duplicate_fen} duplicate FENs."
    )


def build_error_mined_dataset(config: ErrorMineConfig) -> ErrorMineStats:
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = ErrorMineStats()
    seen_fens: set[str] = set()

    con = sqlite3.connect(config.db)
    con.row_factory = sqlite3.Row
    rows = _candidate_rows(con=con, config=config)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            stats.candidates += 1
            fen = str(row["fen"])
            if not row["prompt"]:
                stats.dropped_missing_prompt += 1
                continue
            if row["best_move_uci"] is not None and row["best_move_uci"] == row["accepted_uci"]:
                stats.dropped_best_equals_accepted += 1
                continue
            if fen in seen_fens:
                stats.dropped_duplicate_fen += 1
                continue
            seen_fens.add(fen)
            classification = str(row["classification"])
            run_id = int(row["run_id"])
            payload = {
                "prompt": row["prompt"],
                "completion": (
                    f'{{"move":"{row["best_move_uci"]}"}}'
                    if row["best_move_uci"] is not None
                    else f'{{"move":"{row["accepted_uci"]}"}}'
                ),
                "move": row["best_move_uci"] or row["accepted_uci"],
                "fen": fen,
                "split": "train",
                "source": "arena_error_mined",
                "source_run_id": run_id,
                "source_game_id": row["game_id"],
                "source_move_id": row["move_id"],
                "ply": row["ply"],
                "model_bad_move": row["accepted_uci"],
                "stockfish_best_move": row["best_move_uci"],
                "model_centipawn_loss": row["centipawn_loss"],
                "model_classification": classification,
            }
            output_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stats.written += 1
            stats.by_run[run_id] += 1
            stats.by_classification[classification] += 1
            if config.max_examples is not None and stats.written >= config.max_examples:
                break
    return stats


def _candidate_rows(*, con: sqlite3.Connection, config: ErrorMineConfig) -> list[sqlite3.Row]:
    if not config.run_ids:
        raise ValueError("At least one --run-id is required")
    predicates: list[str] = []
    if config.include_blunders:
        predicates.append("e.classification = 'blunder'")
    if config.include_mate_missed:
        predicates.append("e.classification = 'mate_missed'")
    if config.include_high_mistakes:
        predicates.append("(e.centipawn_loss is not null and e.centipawn_loss >= ?)")
    if not predicates:
        raise ValueError("At least one error type must be enabled")

    run_placeholders = ",".join("?" for _ in config.run_ids)
    sql = f"""
        select
            g.run_id,
            g.id as game_id,
            m.id as move_id,
            m.ply,
            m.fen_before as fen,
            m.accepted_uci,
            e.best_move_uci,
            e.centipawn_loss,
            e.classification,
            (
                select a.raw_prompt
                from attempts a
                where a.move_id = m.id and a.raw_prompt is not null
                order by a.attempt_number
                limit 1
            ) as prompt
        from moves m
        join games g on g.id = m.game_id
        join engine_evaluations e on e.move_id = m.id
        where g.run_id in ({run_placeholders})
          and ({' or '.join(predicates)})
        order by
            case when e.classification = 'blunder' then 0
                 when e.classification = 'mate_missed' then 1
                 else 2
            end,
            coalesce(e.centipawn_loss, 1000000) desc,
            g.run_id,
            g.id,
            m.ply
    """
    params: list[int] = []
    params.extend(config.run_ids)
    if config.include_high_mistakes:
        params.append(config.min_cpl)
    return list(con.execute(sql, params))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GRPO dataset from arena positions where models blundered."
    )
    parser.add_argument("--db", type=Path, default=Path("arena.db"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--run-id", type=int, action="append", required=True)
    parser.add_argument("--min-cpl", type=int, default=250)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--no-blunders", action="store_true")
    parser.add_argument("--no-mate-missed", action="store_true")
    parser.add_argument("--no-high-mistakes", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

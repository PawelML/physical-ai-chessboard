"""Build a broad-plus-tactical JSONL dataset for short GRPO continuation pilots."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MixedTacticalConfig:
    broad_dataset: str
    tactical_dataset: str
    output: str
    metadata_output: str | None
    broad_examples: int
    tactical_repeats: int
    seed: int


@dataclass
class MixedTacticalStats:
    broad_input_rows: int = 0
    tactical_input_rows: int = 0
    output_rows: int = 0
    by_source: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "broad_input_rows": self.broad_input_rows,
            "tactical_input_rows": self.tactical_input_rows,
            "output_rows": self.output_rows,
            "by_source": dict(sorted(self.by_source.items())),
        }


def main() -> None:
    args = _parse_args()
    config = MixedTacticalConfig(
        broad_dataset=str(args.broad_dataset),
        tactical_dataset=str(args.tactical_dataset),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        broad_examples=args.broad_examples,
        tactical_repeats=args.tactical_repeats,
        seed=args.seed,
    )
    stats = build_mixed_tactical_dataset(config)
    if config.metadata_output is not None:
        metadata_path = Path(config.metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {"config": asdict(config), "stats": stats.to_json()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {stats.output_rows} mixed tactical rows to {config.output}.")


def build_mixed_tactical_dataset(config: MixedTacticalConfig) -> MixedTacticalStats:
    rng = random.Random(config.seed)
    broad_rows = _read_jsonl(Path(config.broad_dataset))
    tactical_rows = _read_jsonl(Path(config.tactical_dataset))

    stats = MixedTacticalStats(
        broad_input_rows=len(broad_rows),
        tactical_input_rows=len(tactical_rows),
    )
    broad_count = min(config.broad_examples, len(broad_rows))
    selected_broad = rng.sample(broad_rows, broad_count)

    output_rows: list[dict[str, str]] = []
    for row in selected_broad:
        output_rows.append(_normalize_row(row, source="broad"))
        stats.by_source["broad"] += 1
    for repeat_idx in range(config.tactical_repeats):
        for row in tactical_rows:
            output_rows.append(_normalize_row(row, source=f"tactical_gate_r{repeat_idx + 1}"))
            stats.by_source[f"tactical_gate_r{repeat_idx + 1}"] += 1
    rng.shuffle(output_rows)

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in output_rows:
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")

    stats.output_rows = len(output_rows)
    return stats


def _normalize_row(row: dict[str, Any], *, source: str) -> dict[str, str]:
    return {
        "prompt": str(row["prompt"]),
        "fen": str(row["fen"]),
        "completion": str(row.get("completion") or ""),
        "move": str(row.get("move") or ""),
        "source": source,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a broad-plus-tactical GRPO dataset.")
    parser.add_argument("--broad-dataset", type=Path, required=True)
    parser.add_argument("--tactical-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--broad-examples", type=int, default=3000)
    parser.add_argument("--tactical-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()

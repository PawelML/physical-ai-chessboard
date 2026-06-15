"""Build a held-out tactical evaluation gate from model prediction failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TacticalGateConfig:
    dataset: str
    predictions: list[str]
    output: str
    metadata_output: str | None
    min_cpl: int
    include_blunders: bool
    max_examples: int | None


@dataclass
class TacticalGateStats:
    dataset_rows: int = 0
    prediction_rows: int = 0
    selected_rows: int = 0
    dropped_duplicate_fen: int = 0
    by_source: Counter[str] = field(default_factory=Counter)
    by_classification: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "dataset_rows": self.dataset_rows,
            "prediction_rows": self.prediction_rows,
            "selected_rows": self.selected_rows,
            "dropped_duplicate_fen": self.dropped_duplicate_fen,
            "by_source": dict(sorted(self.by_source.items())),
            "by_classification": dict(sorted(self.by_classification.items())),
        }


def main() -> None:
    args = _parse_args()
    config = TacticalGateConfig(
        dataset=str(args.dataset),
        predictions=[str(path) for path in args.predictions],
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        min_cpl=args.min_cpl,
        include_blunders=not args.no_blunders,
        max_examples=args.max_examples,
    )
    stats = build_tactical_gate(config)
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
    print(f"Wrote {stats.selected_rows} tactical gate rows to {config.output}.")


def build_tactical_gate(config: TacticalGateConfig) -> TacticalGateStats:
    dataset_rows = _read_jsonl(Path(config.dataset))
    by_fen = {str(row["fen"]): row for row in dataset_rows}
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = TacticalGateStats(dataset_rows=len(dataset_rows))
    selected: dict[str, dict[str, Any]] = {}
    for prediction_path in config.predictions:
        source_name = Path(prediction_path).stem
        for prediction in _read_jsonl(Path(prediction_path)):
            stats.prediction_rows += 1
            fen = str(prediction["fen"])
            row = by_fen.get(fen)
            if row is None:
                continue
            classification = str(prediction.get("classification") or "")
            cpl = prediction.get("centipawn_loss")
            high_cpl = isinstance(cpl, int) and cpl >= config.min_cpl
            blunder = config.include_blunders and classification == "blunder"
            if not high_cpl and not blunder:
                continue
            if fen in selected:
                stats.dropped_duplicate_fen += 1
                continue
            gate_row = {
                "prompt": str(row["prompt"]),
                "fen": fen,
                "completion": str(row.get("completion") or ""),
                "move": str(row.get("move") or ""),
                "split": "tactical_gate",
                "gate_source": source_name,
                "gate_trigger_move": str(prediction.get("parsed_move") or ""),
                "gate_trigger_cpl": cpl if isinstance(cpl, int) else None,
                "gate_trigger_classification": classification,
            }
            selected[fen] = gate_row
            stats.by_source[source_name] += 1
            stats.by_classification[classification] += 1
            if config.max_examples is not None and len(selected) >= config.max_examples:
                break
        if config.max_examples is not None and len(selected) >= config.max_examples:
            break

    with output_path.open("w", encoding="utf-8") as output_file:
        for row in selected.values():
            output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
    stats.selected_rows = len(selected)
    return stats


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a held-out tactical gate from prediction JSONL failures."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--min-cpl", type=int, default=250)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--no-blunders", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

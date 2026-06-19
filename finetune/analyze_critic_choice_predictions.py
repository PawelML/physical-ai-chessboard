"""Analyze candidate-choice LoRA predictions against candidate metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    report = analyze_predictions(dataset_path=args.dataset, predictions_path=args.predictions)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def analyze_predictions(*, dataset_path: Path, predictions_path: Path) -> dict[str, Any]:
    rows = _read_jsonl(dataset_path)
    predictions = _read_jsonl(predictions_path)
    if len(rows) != len(predictions):
        raise ValueError(
            f"dataset row count {len(rows)} differs from prediction count {len(predictions)}"
        )

    candidate_ok = 0
    top1_match = 0
    parse_ok = 0
    legal_ok = 0
    selected_cpl: list[int] = []
    target_cpl: list[int] = []
    first_cpl: list[int] = []
    selected_risks: Counter[str] = Counter()
    target_risks: Counter[str] = Counter()
    first_risks: Counter[str] = Counter()
    selected_prompt_index: Counter[int] = Counter()
    target_prompt_index: Counter[int] = Counter()

    for row, prediction in zip(rows, predictions, strict=True):
        parsed_move = prediction.get("parsed_move")
        parse_ok += int(bool(prediction.get("parse_ok")))
        legal_ok += int(bool(prediction.get("legal_ok")))
        top1_match += int(parsed_move == row["move"])

        candidates = list(row["candidates"])
        by_move = {str(candidate["uci"]): candidate for candidate in candidates}
        target = by_move[str(row["move"])]
        first = candidates[0]

        _append_cpl(target_cpl, target)
        _append_cpl(first_cpl, first)
        target_risks[str(target["risk"])] += 1
        first_risks[str(first["risk"])] += 1
        target_prompt_index[int(row["target_prompt_index"])] += 1

        if parsed_move in by_move:
            candidate_ok += 1
            selected = by_move[str(parsed_move)]
            _append_cpl(selected_cpl, selected)
            selected_risks[str(selected["risk"])] += 1
            selected_prompt_index[1 + candidates.index(selected)] += 1

    examples = len(rows)
    return {
        "examples": examples,
        "parse_ok": parse_ok,
        "parse_rate": _rate(parse_ok, examples),
        "legal_ok": legal_ok,
        "legal_rate": _rate(legal_ok, examples),
        "candidate_ok": candidate_ok,
        "candidate_ok_rate": _rate(candidate_ok, examples),
        "top1_match": top1_match,
        "top1_match_rate": _rate(top1_match, examples),
        "selected_mean_centipawn_loss": _mean(selected_cpl),
        "target_oracle_mean_centipawn_loss": _mean(target_cpl),
        "first_prompt_candidate_mean_centipawn_loss": _mean(first_cpl),
        "selected_risk_counts": dict(sorted(selected_risks.items())),
        "target_risk_counts": dict(sorted(target_risks.items())),
        "first_prompt_risk_counts": dict(sorted(first_risks.items())),
        "selected_prompt_index": dict(sorted(selected_prompt_index.items())),
        "target_prompt_index": dict(sorted(target_prompt_index.items())),
    }


def _append_cpl(values: list[int], candidate: dict[str, Any]) -> None:
    cpl = candidate.get("centipawn_loss")
    if isinstance(cpl, int):
        values.append(cpl)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(payload)
    return rows


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate-choice prediction quality.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Analyze candidate-choice LoRA predictions against candidate metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    report = analyze_predictions(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        hard_cases_limit=args.hard_cases_limit if args.hard_cases_output is not None else 0,
        hard_regret_threshold=args.hard_regret_threshold,
    )
    hard_cases = report.pop("hard_cases", [])
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.hard_cases_output is not None:
        _write_jsonl(args.hard_cases_output, hard_cases)


def analyze_predictions(
    *,
    dataset_path: Path,
    predictions_path: Path,
    hard_cases_limit: int = 0,
    hard_regret_threshold: int = 200,
) -> dict[str, Any]:
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
    risk_transitions: Counter[str] = Counter()
    hard_case_counts: Counter[str] = Counter()
    regret_values: list[int] = []
    improved_vs_first = 0
    worsened_vs_first = 0
    tied_first = 0
    hard_cases: list[dict[str, Any]] = []

    for example_index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
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
            risk_transitions[f"{target['risk']}->{selected['risk']}"] += 1
            selected_cpl_value = _candidate_cpl(selected)
            target_cpl_value = _candidate_cpl(target)
            first_cpl_value = _candidate_cpl(first)
            if selected_cpl_value is not None and target_cpl_value is not None:
                regret_values.append(selected_cpl_value - target_cpl_value)
            if selected_cpl_value is not None and first_cpl_value is not None:
                if selected_cpl_value < first_cpl_value:
                    improved_vs_first += 1
                elif selected_cpl_value > first_cpl_value:
                    worsened_vs_first += 1
                else:
                    tied_first += 1
            categories = _hard_case_categories(
                selected=selected,
                target=target,
                hard_regret_threshold=hard_regret_threshold,
            )
            for category in categories:
                hard_case_counts[category] += 1
            if categories and hard_cases_limit > 0:
                hard_cases.append(
                    _hard_case_row(
                        example_index=example_index,
                        row=row,
                        prediction=prediction,
                        categories=categories,
                        selected=selected,
                        target=target,
                        first=first,
                    )
                )
        else:
            hard_case_counts["not_candidate"] += 1
            if hard_cases_limit > 0:
                hard_cases.append(
                    _hard_case_row(
                        example_index=example_index,
                        row=row,
                        prediction=prediction,
                        categories=["not_candidate"],
                        selected=None,
                        target=target,
                        first=first,
                    )
                )

    examples = len(rows)
    hard_cases = sorted(hard_cases, key=_hard_case_sort_key, reverse=True)[:hard_cases_limit]
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
        "risk_transitions": dict(sorted(risk_transitions.items())),
        "selected_high_risk_count": selected_risks["blunder"] + selected_risks["mistake"],
        "mean_cpl_regret_vs_oracle": _mean(regret_values),
        "improved_vs_first_count": improved_vs_first,
        "worsened_vs_first_count": worsened_vs_first,
        "tied_first_count": tied_first,
        "hard_case_counts": dict(sorted(hard_case_counts.items())),
        "hard_cases": hard_cases,
    }


def _append_cpl(values: list[int], candidate: dict[str, Any]) -> None:
    cpl = candidate.get("centipawn_loss")
    if isinstance(cpl, int):
        values.append(cpl)


def _candidate_cpl(candidate: dict[str, Any]) -> int | None:
    cpl = candidate.get("centipawn_loss")
    if isinstance(cpl, int):
        return cpl
    return None


def _hard_case_categories(
    *,
    selected: dict[str, Any],
    target: dict[str, Any],
    hard_regret_threshold: int,
) -> list[str]:
    categories: list[str] = []
    selected_risk = str(selected.get("risk"))
    target_risk = str(target.get("risk"))
    if selected_risk == "blunder":
        categories.append("selected_blunder")
    if selected_risk in {"mistake", "blunder"} and target_risk == "good":
        categories.append("missed_good")
    selected_cpl = _candidate_cpl(selected)
    target_cpl = _candidate_cpl(target)
    if (
        selected_cpl is not None
        and target_cpl is not None
        and selected_cpl - target_cpl >= hard_regret_threshold
    ):
        categories.append("high_regret")
    return categories


def _hard_case_row(
    *,
    example_index: int,
    row: dict[str, Any],
    prediction: dict[str, Any],
    categories: list[str],
    selected: dict[str, Any] | None,
    target: dict[str, Any],
    first: dict[str, Any],
) -> dict[str, Any]:
    selected_cpl = _candidate_cpl(selected) if selected is not None else None
    target_cpl = _candidate_cpl(target)
    first_cpl = _candidate_cpl(first)
    return {
        "example_index": example_index,
        "categories": categories,
        "severity": _hard_case_severity(selected=selected, target=target),
        "fen": row.get("fen") or row.get("fen_before"),
        "side_to_move": row.get("side_to_move"),
        "predicted_move": prediction.get("parsed_move"),
        "target_move": row.get("move") or row.get("target_move"),
        "selected": selected,
        "target": target,
        "first_prompt_candidate": first,
        "cpl_regret_vs_oracle": (
            selected_cpl - target_cpl
            if selected_cpl is not None and target_cpl is not None
            else None
        ),
        "cpl_delta_vs_first": (
            selected_cpl - first_cpl if selected_cpl is not None and first_cpl is not None else None
        ),
        "raw_response": prediction.get("raw_response"),
        "parse_ok": prediction.get("parse_ok"),
        "legal_ok": prediction.get("legal_ok"),
        "candidates": row.get("candidates"),
    }


def _hard_case_severity(
    *,
    selected: dict[str, Any] | None,
    target: dict[str, Any],
) -> int:
    if selected is None:
        return 1_000_000
    selected_cpl = _candidate_cpl(selected)
    target_cpl = _candidate_cpl(target)
    if selected_cpl is None or target_cpl is None:
        return 0
    risk_bonus = 1_000 if selected.get("risk") == "blunder" else 0
    return risk_bonus + selected_cpl - target_cpl


def _hard_case_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    severity = row.get("severity")
    index = row.get("example_index")
    return (
        int(severity) if isinstance(severity, int) else 0,
        -int(index) if isinstance(index, int) else 0,
    )


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


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
    parser.add_argument("--hard-cases-output", type=Path)
    parser.add_argument("--hard-cases-limit", type=int, default=50)
    parser.add_argument("--hard-regret-threshold", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    main()

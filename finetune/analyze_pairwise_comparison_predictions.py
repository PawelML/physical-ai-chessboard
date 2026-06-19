"""Compose pairwise comparison predictions into multi-candidate selections."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from finetune._common import read_jsonl


def main() -> None:
    args = _parse_args()
    report = analyze_pairwise_predictions(
        choice_dataset_path=args.choice_dataset,
        pairwise_dataset_path=args.pairwise_dataset,
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


def analyze_pairwise_predictions(
    *,
    choice_dataset_path: Path,
    pairwise_dataset_path: Path,
    predictions_path: Path,
    hard_cases_limit: int = 0,
    hard_regret_threshold: int = 200,
) -> dict[str, Any]:
    choice_rows = read_jsonl(choice_dataset_path)
    pair_rows = read_jsonl(pairwise_dataset_path)
    predictions = read_jsonl(predictions_path)
    if len(pair_rows) != len(predictions):
        raise ValueError(
            f"pairwise row count {len(pair_rows)} differs from prediction count {len(predictions)}"
        )

    pair_stats = _collect_pair_votes(pair_rows=pair_rows, predictions=predictions)
    evaluated_choice_indices = sorted(pair_stats.votes_by_choice)

    top1_match = 0
    candidate_ok = 0
    selected_cpl: list[int] = []
    target_cpl: list[int] = []
    first_cpl: list[int] = []
    regret_values: list[int] = []
    selected_risks: Counter[str] = Counter()
    target_risks: Counter[str] = Counter()
    first_risks: Counter[str] = Counter()
    selected_prompt_index: Counter[int] = Counter()
    target_prompt_index: Counter[int] = Counter()
    risk_transitions: Counter[str] = Counter()
    hard_case_counts: Counter[str] = Counter()
    improved_vs_first = 0
    worsened_vs_first = 0
    tied_first = 0
    hard_cases: list[dict[str, Any]] = []

    for choice_index in evaluated_choice_indices:
        row = choice_rows[choice_index]
        candidates = list(row["candidates"])
        by_move = {str(candidate["uci"]): candidate for candidate in candidates}
        target = by_move[str(row["move"])]
        first = candidates[0]
        selected_move = _select_by_votes(
            candidates=candidates,
            votes=pair_stats.votes_by_choice[choice_index],
        )
        selected = by_move[selected_move]

        top1_match += int(selected_move == row["move"])
        candidate_ok += 1
        _append_cpl(selected_cpl, selected)
        _append_cpl(target_cpl, target)
        _append_cpl(first_cpl, first)
        selected_risks[str(selected["risk"])] += 1
        target_risks[str(target["risk"])] += 1
        first_risks[str(first["risk"])] += 1
        selected_prompt_index[1 + candidates.index(selected)] += 1
        target_prompt_index[int(row["target_prompt_index"])] += 1
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
                    choice_index=choice_index,
                    row=row,
                    selected=selected,
                    target=target,
                    first=first,
                    votes=pair_stats.votes_by_choice[choice_index],
                    categories=categories,
                )
            )

    examples = len(evaluated_choice_indices)
    hard_cases = sorted(hard_cases, key=_hard_case_sort_key, reverse=True)[:hard_cases_limit]
    return {
        "examples": examples,
        "pair_examples": len(pair_rows),
        "pair_parse_ok": pair_stats.parse_ok,
        "pair_parse_rate": _rate(pair_stats.parse_ok, len(pair_rows)),
        "pair_legal_ok": pair_stats.legal_ok,
        "pair_legal_rate": _rate(pair_stats.legal_ok, len(pair_rows)),
        "pair_candidate_ok": pair_stats.candidate_ok,
        "pair_candidate_ok_rate": _rate(pair_stats.candidate_ok, len(pair_rows)),
        "pair_top1_match": pair_stats.top1_match,
        "pair_top1_match_rate": _rate(pair_stats.top1_match, len(pair_rows)),
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


class PairVoteStats:
    def __init__(self) -> None:
        self.parse_ok = 0
        self.legal_ok = 0
        self.candidate_ok = 0
        self.top1_match = 0
        self.votes_by_choice: defaultdict[int, Counter[str]] = defaultdict(Counter)


def _collect_pair_votes(
    *,
    pair_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> PairVoteStats:
    stats = PairVoteStats()
    for pair_row, prediction in zip(pair_rows, predictions, strict=True):
        choice_index = int(pair_row["pair_source_choice_index"])
        pair_moves = {str(move) for move in pair_row["pair_moves"]}
        parsed_move = prediction.get("parsed_move")
        stats.parse_ok += int(bool(prediction.get("parse_ok")))
        stats.legal_ok += int(bool(prediction.get("legal_ok")))
        stats.top1_match += int(parsed_move == pair_row["move"])
        if parsed_move in pair_moves:
            stats.candidate_ok += 1
            stats.votes_by_choice[choice_index][str(parsed_move)] += 1
        else:
            for move in sorted(pair_moves):
                stats.votes_by_choice[choice_index][move] += 0
    return stats


def _select_by_votes(*, candidates: list[dict[str, Any]], votes: Counter[str]) -> str:
    return max(
        (str(candidate["uci"]) for candidate in candidates),
        key=lambda move: (votes[move], -_candidate_prompt_index(candidates, move)),
    )


def _candidate_prompt_index(candidates: list[dict[str, Any]], move: str) -> int:
    return next(index for index, candidate in enumerate(candidates) if candidate["uci"] == move)


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
    choice_index: int,
    row: dict[str, Any],
    selected: dict[str, Any],
    target: dict[str, Any],
    first: dict[str, Any],
    votes: Counter[str],
    categories: list[str],
) -> dict[str, Any]:
    selected_cpl = _candidate_cpl(selected)
    target_cpl = _candidate_cpl(target)
    first_cpl = _candidate_cpl(first)
    return {
        "choice_index": choice_index,
        "categories": categories,
        "severity": _hard_case_severity(selected=selected, target=target),
        "fen": row.get("fen") or row.get("fen_before"),
        "side_to_move": row.get("side_to_move"),
        "selected_move": selected.get("uci"),
        "target_move": row.get("move") or row.get("target_move"),
        "selected": selected,
        "target": target,
        "first_prompt_candidate": first,
        "pairwise_votes": dict(sorted(votes.items())),
        "cpl_regret_vs_oracle": (
            selected_cpl - target_cpl
            if selected_cpl is not None and target_cpl is not None
            else None
        ),
        "cpl_delta_vs_first": (
            selected_cpl - first_cpl if selected_cpl is not None and first_cpl is not None else None
        ),
        "candidates": row.get("candidates"),
    }


def _hard_case_severity(
    *,
    selected: dict[str, Any],
    target: dict[str, Any],
) -> int:
    selected_cpl = _candidate_cpl(selected)
    target_cpl = _candidate_cpl(target)
    if selected_cpl is None or target_cpl is None:
        return 0
    risk_bonus = 1_000 if selected.get("risk") == "blunder" else 0
    return risk_bonus + selected_cpl - target_cpl


def _hard_case_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    severity = row.get("severity")
    index = row.get("choice_index")
    return (
        int(severity) if isinstance(severity, int) else 0,
        -(int(index) if isinstance(index, int) else 0),
    )


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose pairwise comparison predictions into multi-candidate selections."
    )
    parser.add_argument("--choice-dataset", type=Path, required=True)
    parser.add_argument("--pairwise-dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hard-cases-output", type=Path)
    parser.add_argument("--hard-cases-limit", type=int, default=100)
    parser.add_argument("--hard-regret-threshold", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    main()

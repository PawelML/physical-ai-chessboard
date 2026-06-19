"""Analyze whether critic-ranker JSONL rows contain a useful ranking signal."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_RISKS = {"good", "playable"}
UNSAFE_RISKS = {"mistake", "blunder"}
GENERATOR_CANDIDATE_SOURCES = {
    "arena_candidate",
    "arena_candidate_pairwise",
    "policy_sample",
}


@dataclass(frozen=True)
class CandidateRow:
    fen_before: str
    candidate_uci: str
    source: str
    risk: str
    score: int
    centipawn_loss: int | None
    candidate_rank_in_generator: int | None


def main() -> None:
    args = _parse_args()
    report = analyze_dataset(args.dataset)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def analyze_dataset(path: Path) -> dict[str, Any]:
    rows = list(_read_rows(path))
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    for row in rows:
        grouped[row.fen_before].append(row)
        source_counts[row.source] += 1
        risk_counts[row.risk] += 1

    final_rows: list[CandidateRow] = []
    first_generator_rows: list[CandidateRow] = []
    oracle_rows: list[CandidateRow] = []
    final_oracle_pairs: list[tuple[CandidateRow, CandidateRow]] = []
    first_generator_oracle_pairs: list[tuple[CandidateRow, CandidateRow]] = []
    mixed_positions = 0
    qwen_candidate_positions = 0
    final_blunder_oracle_safe = 0
    first_generator_blunder_oracle_safe = 0

    for candidates in grouped.values():
        risks = {candidate.risk for candidate in candidates}
        if risks & SAFE_RISKS and risks & UNSAFE_RISKS:
            mixed_positions += 1

        final = _arena_final(candidates)
        first_generator = _first_generator_candidate(candidates)
        oracle = _oracle_candidate(candidates)

        if final is not None:
            final_rows.append(final)
        if first_generator is not None:
            qwen_candidate_positions += 1
            first_generator_rows.append(first_generator)
        if oracle is not None:
            oracle_rows.append(oracle)
        if final is not None and oracle is not None:
            final_oracle_pairs.append((final, oracle))
            if final.risk == "blunder" and oracle.risk in SAFE_RISKS:
                final_blunder_oracle_safe += 1
        if first_generator is not None and oracle is not None:
            first_generator_oracle_pairs.append((first_generator, oracle))
            if first_generator.risk == "blunder" and oracle.risk in SAFE_RISKS:
                first_generator_blunder_oracle_safe += 1

    return {
        "dataset": str(path),
        "rows": len(rows),
        "positions": len(grouped),
        "positions_with_qwen_candidates": qwen_candidate_positions,
        "mixed_safe_unsafe_positions": mixed_positions,
        "risk_counts": dict(sorted(risk_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "candidates_per_position": _count_summary(
            [len(candidates) for candidates in grouped.values()]
        ),
        "arena_final": _selection_summary(final_rows),
        "first_generator_candidate": _selection_summary(first_generator_rows),
        "oracle_candidate": _selection_summary(oracle_rows),
        "oracle_gain_vs_arena_final": _gain_summary(final_oracle_pairs),
        "oracle_gain_vs_first_generator": _gain_summary(first_generator_oracle_pairs),
        "final_blunder_oracle_safe": final_blunder_oracle_safe,
        "first_generator_blunder_oracle_safe": first_generator_blunder_oracle_safe,
    }


def _read_rows(path: Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(_candidate_from_payload(payload, line_number=line_number))
    return rows


def _candidate_from_payload(payload: dict[str, Any], *, line_number: int) -> CandidateRow:
    required = ["fen_before", "candidate_uci", "source", "risk", "score"]
    missing = [field for field in required if field not in payload]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"line {line_number} missing required fields: {missing_text}")

    score = payload["score"]
    if not isinstance(score, int):
        raise ValueError(f"line {line_number} has non-integer score")

    centipawn_loss = payload.get("centipawn_loss")
    if centipawn_loss is not None and not isinstance(centipawn_loss, int):
        raise ValueError(f"line {line_number} has non-integer centipawn_loss")

    rank = payload.get("candidate_rank_in_generator")
    if rank is not None and not isinstance(rank, int):
        raise ValueError(f"line {line_number} has non-integer candidate_rank_in_generator")

    return CandidateRow(
        fen_before=str(payload["fen_before"]),
        candidate_uci=str(payload["candidate_uci"]),
        source=str(payload["source"]),
        risk=str(payload["risk"]),
        score=score,
        centipawn_loss=centipawn_loss,
        candidate_rank_in_generator=rank,
    )


def _arena_final(candidates: list[CandidateRow]) -> CandidateRow | None:
    for candidate in candidates:
        if candidate.source in {"arena_move", "arena_blunder"}:
            return candidate
    return None


def _first_generator_candidate(candidates: list[CandidateRow]) -> CandidateRow | None:
    generator_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate.source in GENERATOR_CANDIDATE_SOURCES
            and candidate.candidate_rank_in_generator is not None
        )
    ]
    if not generator_candidates:
        return None
    return min(
        generator_candidates,
        key=lambda candidate: candidate.candidate_rank_in_generator or 0,
    )


def _oracle_candidate(candidates: list[CandidateRow]) -> CandidateRow | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.score, _negative_cpl(candidate)))


def _negative_cpl(candidate: CandidateRow) -> int:
    if candidate.centipawn_loss is None:
        return -999_999
    return -candidate.centipawn_loss


def _selection_summary(rows: list[CandidateRow]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_score": _mean([row.score for row in rows]),
        "mean_centipawn_loss": _mean(
            [row.centipawn_loss for row in rows if row.centipawn_loss is not None]
        ),
        "risk_counts": dict(sorted(Counter(row.risk for row in rows).items())),
    }


def _gain_summary(paired: list[tuple[CandidateRow, CandidateRow]]) -> dict[str, Any]:
    score_gains = [oracle.score - baseline.score for baseline, oracle in paired]
    cpl_gains = [
        baseline.centipawn_loss - oracle.centipawn_loss
        for baseline, oracle in paired
        if baseline.centipawn_loss is not None and oracle.centipawn_loss is not None
    ]
    return {
        "positions": len(paired),
        "mean_score_gain": _mean(score_gains),
        "mean_centipawn_loss_reduction": _mean(cpl_gains),
        "improved_positions_by_score": sum(gain > 0 for gain in score_gains),
        "improved_positions_by_cpl": sum(gain > 0 for gain in cpl_gains),
    }


def _count_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": _mean(values)}


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze critic-ranker JSONL dataset quality.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()

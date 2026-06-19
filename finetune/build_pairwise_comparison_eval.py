"""Expand multi-candidate choice rows into all pairwise comparison prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from finetune._common import read_jsonl, write_metadata_sidecar
from finetune.build_critic_choice_dataset import CandidateChoice
from finetune.build_critic_pairwise_dataset import build_pairwise_prompt


@dataclass(frozen=True)
class PairwiseComparisonEvalConfig:
    input: str
    output: str
    metadata_output: str | None
    max_choice_rows: int | None = None
    max_pair_rows: int | None = None


@dataclass
class PairwiseComparisonEvalStats:
    choice_rows_seen: int = 0
    choice_rows_written: int = 0
    pair_rows_written: int = 0
    skipped_too_few_candidates: int = 0
    by_split: Counter[str] = field(default_factory=Counter)
    by_choice_candidate_count: Counter[int] = field(default_factory=Counter)
    by_pair_target_prompt_index: Counter[int] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "choice_rows_seen": self.choice_rows_seen,
            "choice_rows_written": self.choice_rows_written,
            "pair_rows_written": self.pair_rows_written,
            "skipped_too_few_candidates": self.skipped_too_few_candidates,
            "by_split": dict(sorted(self.by_split.items())),
            "by_choice_candidate_count": dict(sorted(self.by_choice_candidate_count.items())),
            "by_pair_target_prompt_index": dict(sorted(self.by_pair_target_prompt_index.items())),
        }


def main() -> None:
    args = _parse_args()
    config = PairwiseComparisonEvalConfig(
        input=str(args.input),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        max_choice_rows=args.max_choice_rows,
        max_pair_rows=args.max_pair_rows,
    )
    stats = build_pairwise_comparison_eval(config)
    if config.metadata_output is not None:
        write_metadata_sidecar(Path(config.metadata_output), config=config, stats=stats)
    print(
        f"Wrote {stats.pair_rows_written} pairwise comparison rows from "
        f"{stats.choice_rows_written}/{stats.choice_rows_seen} choice rows to {config.output}."
    )


def build_pairwise_comparison_eval(
    config: PairwiseComparisonEvalConfig,
) -> PairwiseComparisonEvalStats:
    stats = PairwiseComparisonEvalStats()
    rows = read_jsonl(Path(config.input))
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for choice_index, choice_row in enumerate(rows):
            if (
                config.max_choice_rows is not None
                and stats.choice_rows_seen >= config.max_choice_rows
            ):
                break
            stats.choice_rows_seen += 1
            pair_rows = _pair_rows_for_choice(choice_index=choice_index, choice_row=choice_row)
            if not pair_rows:
                stats.skipped_too_few_candidates += 1
                continue
            stats.choice_rows_written += 1
            stats.by_split[str(choice_row.get("split", "unknown"))] += 1
            stats.by_choice_candidate_count[int(choice_row["candidate_count"])] += 1
            for pair_row in pair_rows:
                if (
                    config.max_pair_rows is not None
                    and stats.pair_rows_written >= config.max_pair_rows
                ):
                    return stats
                output_file.write(json.dumps(pair_row, separators=(",", ":")) + "\n")
                stats.pair_rows_written += 1
                stats.by_pair_target_prompt_index[int(pair_row["target_prompt_index"])] += 1
    return stats


def _pair_rows_for_choice(*, choice_index: int, choice_row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [_candidate_from_choice(candidate) for candidate in choice_row["candidates"]]
    if len(candidates) < 2:
        return []
    rows: list[dict[str, Any]] = []
    for pair_index, (left, right) in enumerate(combinations(candidates, 2)):
        better, worse = sorted([left, right], key=_candidate_quality_key, reverse=True)
        visible = sorted(
            [better, worse],
            key=lambda candidate: _prompt_order_key(
                choice_index=choice_index,
                pair_index=pair_index,
                candidate=candidate,
            ),
        )
        target_prompt_index = 1 + next(
            index for index, candidate in enumerate(visible) if candidate.uci == better.uci
        )
        rows.append(
            {
                "prompt": build_pairwise_prompt(
                    fen=str(choice_row["fen"]),
                    side_to_move=str(choice_row.get("side_to_move", "unknown")),
                    candidates=visible,
                ),
                "completion": json.dumps({"move": better.uci}, separators=(",", ":")),
                "fen": str(choice_row["fen"]),
                "move": better.uci,
                "fen_before": str(choice_row.get("fen_before", choice_row["fen"])),
                "side_to_move": str(choice_row.get("side_to_move", "unknown")),
                "split": str(choice_row.get("split", "unknown")),
                "candidate_count": 2,
                "candidates": [asdict(candidate) for candidate in visible],
                "target_move": better.uci,
                "target_prompt_index": target_prompt_index,
                "target_risk": better.risk,
                "target_score": better.score,
                "target_centipawn_loss": better.centipawn_loss,
                "pair_source_choice_index": choice_index,
                "pair_index": pair_index,
                "pair_moves": [left.uci, right.uci],
                "source_choice_move": str(choice_row["move"]),
                "source_choice_candidate_count": int(choice_row["candidate_count"]),
            }
        )
    return rows


def _candidate_from_choice(candidate: dict[str, Any]) -> CandidateChoice:
    return CandidateChoice(
        uci=str(candidate["uci"]),
        san=str(candidate["san"]),
        source=str(candidate["source"]),
        risk=str(candidate["risk"]),
        score=int(candidate["score"]),
        centipawn_loss=(
            int(candidate["centipawn_loss"])
            if candidate.get("centipawn_loss") is not None
            else None
        ),
        candidate_rank_in_generator=(
            int(candidate["candidate_rank_in_generator"])
            if candidate.get("candidate_rank_in_generator") is not None
            else None
        ),
    )


def _candidate_quality_key(candidate: CandidateChoice) -> tuple[int, int, int]:
    cpl_key = -999_999 if candidate.centipawn_loss is None else -candidate.centipawn_loss
    generator_rank_key = (
        -999_999
        if candidate.candidate_rank_in_generator is None
        else -candidate.candidate_rank_in_generator
    )
    return candidate.score, cpl_key, generator_rank_key


def _prompt_order_key(*, choice_index: int, pair_index: int, candidate: CandidateChoice) -> str:
    return hashlib.sha256(f"{choice_index}|{pair_index}|{candidate.uci}".encode()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand multi-candidate choice rows into pairwise comparison prompts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--max-choice-rows", type=int)
    parser.add_argument("--max-pair-rows", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()

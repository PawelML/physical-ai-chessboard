"""Build pairwise safe-vs-unsafe candidate choice examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finetune._common import read_jsonl, write_metadata_sidecar
from finetune.build_critic_choice_dataset import SAFE_RISKS, UNSAFE_RISKS, CandidateChoice
from finetune.build_critic_ranker_dataset import split_for_fen


@dataclass(frozen=True)
class PairwiseDatasetConfig:
    input: str
    output: str
    metadata_output: str | None
    pairs_per_position: int = 4
    min_cpl_gap: int = 100
    max_positions: int | None = None
    max_examples: int | None = None


@dataclass
class PairwiseDatasetStats:
    input_rows: int = 0
    positions_seen: int = 0
    positions_with_pairs: int = 0
    examples_written: int = 0
    dropped_no_safe_candidate: int = 0
    dropped_no_unsafe_candidate: int = 0
    dropped_small_gap_pairs: int = 0
    dropped_pair_limit: int = 0
    by_split: Counter[str] = field(default_factory=Counter)
    by_target_risk: Counter[str] = field(default_factory=Counter)
    by_pair_risks: Counter[str] = field(default_factory=Counter)
    by_target_prompt_index: Counter[int] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "positions_seen": self.positions_seen,
            "positions_with_pairs": self.positions_with_pairs,
            "examples_written": self.examples_written,
            "dropped_no_safe_candidate": self.dropped_no_safe_candidate,
            "dropped_no_unsafe_candidate": self.dropped_no_unsafe_candidate,
            "dropped_small_gap_pairs": self.dropped_small_gap_pairs,
            "dropped_pair_limit": self.dropped_pair_limit,
            "by_split": dict(sorted(self.by_split.items())),
            "by_target_risk": dict(sorted(self.by_target_risk.items())),
            "by_pair_risks": dict(sorted(self.by_pair_risks.items())),
            "by_target_prompt_index": dict(sorted(self.by_target_prompt_index.items())),
        }


@dataclass(frozen=True)
class CandidatePair:
    better: CandidateChoice
    worse: CandidateChoice
    gap: int | None


def main() -> None:
    args = _parse_args()
    config = PairwiseDatasetConfig(
        input=str(args.input),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        pairs_per_position=args.pairs_per_position,
        min_cpl_gap=args.min_cpl_gap,
        max_positions=args.max_positions,
        max_examples=args.max_examples,
    )
    stats = build_pairwise_dataset(config)
    if config.metadata_output is not None:
        write_metadata_sidecar(Path(config.metadata_output), config=config, stats=stats)
    print(
        f"Wrote {stats.examples_written} pairwise rows from "
        f"{stats.positions_with_pairs}/{stats.positions_seen} positions to {config.output}."
    )


def build_pairwise_dataset(config: PairwiseDatasetConfig) -> PairwiseDatasetStats:
    stats = PairwiseDatasetStats()
    grouped: dict[str, list[CandidateChoice]] = defaultdict(list)
    side_by_fen: dict[str, str] = {}
    for input_row in read_jsonl(Path(config.input)):
        stats.input_rows += 1
        fen = str(input_row["fen_before"])
        grouped[fen].append(_candidate_from_row(input_row))
        side_by_fen[fen] = str(input_row.get("side_to_move", "unknown"))

    stats.positions_seen = len(grouped)
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for fen, candidates in grouped.items():
            if (
                config.max_positions is not None
                and stats.positions_with_pairs >= config.max_positions
            ):
                break
            pairs = _candidate_pairs(
                fen=fen,
                candidates=_dedupe_candidates(candidates),
                config=config,
                stats=stats,
            )
            if not pairs:
                continue
            stats.positions_with_pairs += 1
            for pair_index, pair in enumerate(pairs):
                if (
                    config.max_examples is not None
                    and stats.examples_written >= config.max_examples
                ):
                    return stats
                row = _build_pair_row(
                    fen=fen,
                    side_to_move=side_by_fen[fen],
                    pair=pair,
                    pair_index=pair_index,
                )
                output_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                stats.examples_written += 1
                stats.by_split[str(row["split"])] += 1
                stats.by_target_risk[str(row["target_risk"])] += 1
                stats.by_pair_risks[str(row["pair_risks"])] += 1
                stats.by_target_prompt_index[int(row["target_prompt_index"])] += 1
    return stats


def build_pairwise_prompt(
    *,
    fen: str,
    side_to_move: str,
    candidates: list[CandidateChoice],
) -> str:
    candidate_lines = [
        f"{index}. {candidate.uci} ({candidate.san})"
        for index, candidate in enumerate(candidates, start=1)
    ]
    return "\n".join(
        [
            "You are a chess move safety comparator.",
            "",
            "Position:",
            f"FEN: {fen}",
            f"Side to move: {side_to_move}",
            "",
            "Candidate moves:",
            *candidate_lines,
            "",
            "Choose the safer move from the two candidates.",
            "Return JSON only:",
            '{"move":"<uci>"}',
        ]
    )


def _candidate_pairs(
    *,
    fen: str,
    candidates: list[CandidateChoice],
    config: PairwiseDatasetConfig,
    stats: PairwiseDatasetStats,
) -> list[CandidatePair]:
    safe = [candidate for candidate in candidates if candidate.risk in SAFE_RISKS]
    unsafe = [candidate for candidate in candidates if candidate.risk in UNSAFE_RISKS]
    if not safe:
        stats.dropped_no_safe_candidate += 1
        return []
    if not unsafe:
        stats.dropped_no_unsafe_candidate += 1
        return []

    pairs: list[CandidatePair] = []
    for safe_candidate in safe:
        for unsafe_candidate in unsafe:
            gap = _cpl_gap(safe_candidate, unsafe_candidate)
            if gap is not None and gap < config.min_cpl_gap:
                stats.dropped_small_gap_pairs += 1
                continue
            better, worse = sorted(
                [safe_candidate, unsafe_candidate],
                key=_candidate_quality_key,
                reverse=True,
            )
            pairs.append(CandidatePair(better=better, worse=worse, gap=gap))

    pairs = sorted(pairs, key=lambda pair: _pair_sort_key(fen=fen, pair=pair), reverse=True)
    if len(pairs) > config.pairs_per_position:
        stats.dropped_pair_limit += len(pairs) - config.pairs_per_position
    return pairs[: config.pairs_per_position]


def _build_pair_row(
    *,
    fen: str,
    side_to_move: str,
    pair: CandidatePair,
    pair_index: int,
) -> dict[str, Any]:
    visible = sorted(
        [pair.better, pair.worse],
        key=lambda candidate: _prompt_order_key(
            fen=fen,
            pair_index=pair_index,
            candidate=candidate,
        ),
    )
    target_prompt_index = 1 + next(
        index for index, candidate in enumerate(visible) if candidate.uci == pair.better.uci
    )
    completion = json.dumps({"move": pair.better.uci}, separators=(",", ":"))
    return {
        "prompt": build_pairwise_prompt(fen=fen, side_to_move=side_to_move, candidates=visible),
        "completion": completion,
        "fen": fen,
        "move": pair.better.uci,
        "fen_before": fen,
        "side_to_move": side_to_move,
        "split": split_for_fen(fen),
        "candidate_count": 2,
        "candidates": [asdict(candidate) for candidate in visible],
        "target_move": pair.better.uci,
        "target_san": pair.better.san,
        "target_prompt_index": target_prompt_index,
        "target_risk": pair.better.risk,
        "target_score": pair.better.score,
        "target_centipawn_loss": pair.better.centipawn_loss,
        "pair_risks": _pair_risks(pair.better, pair.worse),
        "pair_cpl_gap": pair.gap,
    }


def _dedupe_candidates(candidates: list[CandidateChoice]) -> list[CandidateChoice]:
    result: dict[str, CandidateChoice] = {}
    for candidate in candidates:
        current = result.get(candidate.uci)
        if current is None or _candidate_quality_key(candidate) > _candidate_quality_key(current):
            result[candidate.uci] = candidate
    return list(result.values())


def _candidate_from_row(row: dict[str, Any]) -> CandidateChoice:
    return CandidateChoice(
        uci=str(row["candidate_uci"]),
        san=str(row["candidate_san"]),
        source=str(row["source"]),
        risk=str(row["risk"]),
        score=int(row["score"]),
        centipawn_loss=(
            int(row["centipawn_loss"]) if row.get("centipawn_loss") is not None else None
        ),
        candidate_rank_in_generator=(
            int(row["candidate_rank_in_generator"])
            if row.get("candidate_rank_in_generator") is not None
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


def _cpl_gap(left: CandidateChoice, right: CandidateChoice) -> int | None:
    if left.centipawn_loss is None or right.centipawn_loss is None:
        return None
    return abs(left.centipawn_loss - right.centipawn_loss)


def _pair_sort_key(*, fen: str, pair: CandidatePair) -> tuple[int, str]:
    gap = -1 if pair.gap is None else pair.gap
    return gap, hashlib.sha256(f"{fen}|{pair.better.uci}|{pair.worse.uci}".encode()).hexdigest()


def _prompt_order_key(*, fen: str, pair_index: int, candidate: CandidateChoice) -> str:
    return hashlib.sha256(f"{fen}|{pair_index}|{candidate.uci}".encode()).hexdigest()


def _pair_risks(better: CandidateChoice, worse: CandidateChoice) -> str:
    return "->".join(sorted([better.risk, worse.risk]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pairwise safe-vs-unsafe candidate choice examples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--pairs-per-position", type=int, default=4)
    parser.add_argument("--min-cpl-gap", type=int, default=100)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Build position-level choice examples from candidate-level critic rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finetune.build_critic_ranker_dataset import split_for_fen

SAFE_RISKS = {"good", "playable"}
UNSAFE_RISKS = {"mistake", "blunder"}


@dataclass(frozen=True)
class ChoiceDatasetConfig:
    input: str
    output: str
    metadata_output: str | None
    min_candidates: int = 2
    max_candidates: int = 8
    require_mixed_risk: bool = True
    max_positions: int | None = None


@dataclass(frozen=True)
class CandidateChoice:
    uci: str
    san: str
    source: str
    risk: str
    score: int
    centipawn_loss: int | None
    candidate_rank_in_generator: int | None


@dataclass
class ChoiceDatasetStats:
    input_rows: int = 0
    positions_seen: int = 0
    positions_written: int = 0
    dropped_too_few_candidates: int = 0
    dropped_no_mixed_risk: int = 0
    dropped_candidate_limit: int = 0
    by_split: Counter[str] = field(default_factory=Counter)
    by_target_risk: Counter[str] = field(default_factory=Counter)
    by_candidate_count: Counter[int] = field(default_factory=Counter)
    by_target_prompt_index: Counter[int] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "positions_seen": self.positions_seen,
            "positions_written": self.positions_written,
            "dropped_too_few_candidates": self.dropped_too_few_candidates,
            "dropped_no_mixed_risk": self.dropped_no_mixed_risk,
            "dropped_candidate_limit": self.dropped_candidate_limit,
            "by_split": dict(sorted(self.by_split.items())),
            "by_target_risk": dict(sorted(self.by_target_risk.items())),
            "by_candidate_count": dict(sorted(self.by_candidate_count.items())),
            "by_target_prompt_index": dict(sorted(self.by_target_prompt_index.items())),
        }


def main() -> None:
    args = _parse_args()
    config = ChoiceDatasetConfig(
        input=str(args.input),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        require_mixed_risk=args.allow_non_mixed is False,
        max_positions=args.max_positions,
    )
    stats = build_choice_dataset(config)
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
    print(
        f"Wrote {stats.positions_written} choice rows from "
        f"{stats.positions_seen} positions to {config.output}."
    )


def build_choice_dataset(config: ChoiceDatasetConfig) -> ChoiceDatasetStats:
    stats = ChoiceDatasetStats()
    grouped: dict[str, list[CandidateChoice]] = defaultdict(list)
    side_by_fen: dict[str, str] = {}
    for input_row in _read_jsonl(Path(config.input)):
        stats.input_rows += 1
        fen = str(input_row["fen_before"])
        grouped[fen].append(_candidate_from_row(input_row))
        side_by_fen[fen] = str(input_row.get("side_to_move", "unknown"))

    stats.positions_seen = len(grouped)
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for fen, candidates in grouped.items():
            if config.max_positions is not None and stats.positions_written >= config.max_positions:
                break
            output_row = _build_choice_row(
                fen=fen,
                side_to_move=side_by_fen[fen],
                candidates=candidates,
                config=config,
                stats=stats,
            )
            if output_row is None:
                continue
            output_file.write(json.dumps(output_row, separators=(",", ":")) + "\n")
            stats.positions_written += 1
            stats.by_split[str(output_row["split"])] += 1
            stats.by_target_risk[str(output_row["target_risk"])] += 1
            stats.by_candidate_count[int(output_row["candidate_count"])] += 1
            stats.by_target_prompt_index[int(output_row["target_prompt_index"])] += 1
    return stats


def _build_choice_row(
    *,
    fen: str,
    side_to_move: str,
    candidates: list[CandidateChoice],
    config: ChoiceDatasetConfig,
    stats: ChoiceDatasetStats,
) -> dict[str, Any] | None:
    deduped = _dedupe_candidates(candidates)
    if len(deduped) < config.min_candidates:
        stats.dropped_too_few_candidates += 1
        return None
    if config.require_mixed_risk:
        risks = {candidate.risk for candidate in deduped}
        if not (risks & SAFE_RISKS and risks & UNSAFE_RISKS):
            stats.dropped_no_mixed_risk += 1
            return None
    ranked = sorted(deduped, key=_candidate_sort_key, reverse=True)
    if len(ranked) > config.max_candidates:
        stats.dropped_candidate_limit += len(ranked) - config.max_candidates
    selected = ranked[: config.max_candidates]
    target = selected[0]
    visible = sorted(selected, key=lambda candidate: _prompt_order_key(fen, candidate))
    target_prompt_index = 1 + next(
        index for index, candidate in enumerate(visible) if candidate.uci == target.uci
    )
    completion = json.dumps({"move": target.uci}, separators=(",", ":"))
    return {
        "prompt": build_choice_prompt(
            fen=fen,
            side_to_move=side_to_move,
            candidates=visible,
        ),
        "completion": completion,
        "fen": fen,
        "move": target.uci,
        "fen_before": fen,
        "side_to_move": side_to_move,
        "split": split_for_fen(fen),
        "candidate_count": len(visible),
        "candidates": [asdict(candidate) for candidate in visible],
        "target_move": target.uci,
        "target_san": target.san,
        "target_prompt_index": target_prompt_index,
        "target_risk": target.risk,
        "target_score": target.score,
        "target_centipawn_loss": target.centipawn_loss,
    }


def build_choice_prompt(
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
            "You are a chess candidate move ranker.",
            "",
            "Position:",
            f"FEN: {fen}",
            f"Side to move: {side_to_move}",
            "",
            "Candidate moves:",
            *candidate_lines,
            "",
            "Choose the best move from the candidate list.",
            "Return JSON only:",
            '{"move":"<uci>"}',
        ]
    )


def _dedupe_candidates(candidates: list[CandidateChoice]) -> list[CandidateChoice]:
    result: dict[str, CandidateChoice] = {}
    for candidate in candidates:
        current = result.get(candidate.uci)
        if current is None or _candidate_sort_key(candidate) > _candidate_sort_key(current):
            result[candidate.uci] = candidate
    return list(result.values())


def _candidate_sort_key(candidate: CandidateChoice) -> tuple[int, int, int]:
    cpl_key = -999_999 if candidate.centipawn_loss is None else -candidate.centipawn_loss
    generator_rank_key = (
        -999_999
        if candidate.candidate_rank_in_generator is None
        else -candidate.candidate_rank_in_generator
    )
    return candidate.score, cpl_key, generator_rank_key


def _prompt_order_key(fen: str, candidate: CandidateChoice) -> str:
    return hashlib.sha256(f"{fen}|{candidate.uci}".encode()).hexdigest()


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build position-level best-candidate choice examples."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--min-candidates", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--allow-non-mixed", action="store_true")
    parser.add_argument("--max-positions", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()

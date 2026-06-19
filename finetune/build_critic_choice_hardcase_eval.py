"""Build a regression eval dataset from critic-choice hard cases."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from finetune._common import write_metadata_sidecar
from finetune.build_critic_choice_dataset import CandidateChoice, build_choice_prompt


@dataclass(frozen=True)
class HardcaseEvalConfig:
    input: str
    output: str
    metadata_output: str | None
    variants_per_case: int = 1
    seed: int = 0
    max_cases: int | None = None


@dataclass
class HardcaseEvalStats:
    input_hard_cases: int = 0
    rows_written: int = 0
    dropped_missing_target: int = 0
    dropped_too_few_candidates: int = 0
    by_category: Counter[str] = field(default_factory=Counter)
    by_target_risk: Counter[str] = field(default_factory=Counter)

    def to_json(self) -> dict[str, Any]:
        return {
            "input_hard_cases": self.input_hard_cases,
            "rows_written": self.rows_written,
            "dropped_missing_target": self.dropped_missing_target,
            "dropped_too_few_candidates": self.dropped_too_few_candidates,
            "by_category": dict(sorted(self.by_category.items())),
            "by_target_risk": dict(sorted(self.by_target_risk.items())),
        }


def main() -> None:
    args = _parse_args()
    config = HardcaseEvalConfig(
        input=str(args.input),
        output=str(args.output),
        metadata_output=str(args.metadata_output) if args.metadata_output else None,
        variants_per_case=args.variants_per_case,
        seed=args.seed,
        max_cases=args.max_cases,
    )
    stats = build_hardcase_eval_dataset(config)
    if config.metadata_output is not None:
        write_metadata_sidecar(Path(config.metadata_output), config=config, stats=stats)
    print(f"Wrote {stats.rows_written} hard-case eval rows to {config.output}.")


def build_hardcase_eval_dataset(config: HardcaseEvalConfig) -> HardcaseEvalStats:
    stats = HardcaseEvalStats()
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variants_per_case = max(config.variants_per_case, 1)

    with output_path.open("w", encoding="utf-8") as output_file:
        for hard_case in _read_jsonl(Path(config.input)):
            if config.max_cases is not None and stats.input_hard_cases >= config.max_cases:
                break
            stats.input_hard_cases += 1
            candidates = [
                _candidate_from_payload(candidate) for candidate in hard_case["candidates"]
            ]
            if len(candidates) < 2:
                stats.dropped_too_few_candidates += 1
                continue

            target_uci = str(hard_case["target"]["uci"])
            target = next(
                (candidate for candidate in candidates if candidate.uci == target_uci),
                None,
            )
            if target is None:
                stats.dropped_missing_target += 1
                continue

            for category in hard_case.get("categories", []):
                stats.by_category[str(category)] += 1
            stats.by_target_risk[target.risk] += 1

            for variant_index in range(variants_per_case):
                visible = list(candidates)
                if variant_index > 0:
                    random.Random(_variant_seed(config, hard_case, variant_index)).shuffle(visible)
                output_file.write(
                    json.dumps(
                        _build_output_row(
                            hard_case=hard_case,
                            candidates=visible,
                            target=target,
                            variant_index=variant_index,
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stats.rows_written += 1
    return stats


def _build_output_row(
    *,
    hard_case: dict[str, Any],
    candidates: list[CandidateChoice],
    target: CandidateChoice,
    variant_index: int,
) -> dict[str, Any]:
    target_prompt_index = 1 + next(
        index for index, candidate in enumerate(candidates) if candidate.uci == target.uci
    )
    fen = str(hard_case["fen"])
    side_to_move = str(hard_case.get("side_to_move", "unknown"))
    completion = json.dumps({"move": target.uci}, separators=(",", ":"))
    return {
        "prompt": build_choice_prompt(
            fen=fen,
            side_to_move=side_to_move,
            candidates=candidates,
        ),
        "completion": completion,
        "fen": fen,
        "move": target.uci,
        "fen_before": fen,
        "side_to_move": side_to_move,
        "split": "hardcase_eval",
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "target_move": target.uci,
        "target_san": target.san,
        "target_prompt_index": target_prompt_index,
        "target_risk": target.risk,
        "target_score": target.score,
        "target_centipawn_loss": target.centipawn_loss,
        "hard_case_categories": hard_case.get("categories", []),
        "hard_case_example_index": hard_case.get("example_index"),
        "hard_case_variant_index": variant_index,
        "hard_case_selected_move": hard_case.get("predicted_move"),
        "hard_case_cpl_regret_vs_oracle": hard_case.get("cpl_regret_vs_oracle"),
        "hard_case_cpl_delta_vs_first": hard_case.get("cpl_delta_vs_first"),
    }


def _candidate_from_payload(payload: dict[str, Any]) -> CandidateChoice:
    return CandidateChoice(
        uci=str(payload["uci"]),
        san=str(payload["san"]),
        source=str(payload["source"]),
        risk=str(payload["risk"]),
        score=int(payload["score"]),
        centipawn_loss=(
            int(payload["centipawn_loss"]) if payload.get("centipawn_loss") is not None else None
        ),
        candidate_rank_in_generator=(
            int(payload["candidate_rank_in_generator"])
            if payload.get("candidate_rank_in_generator") is not None
            else None
        ),
    )


def _variant_seed(
    config: HardcaseEvalConfig,
    hard_case: dict[str, Any],
    variant_index: int,
) -> str:
    return f"{config.seed}|{hard_case.get('example_index')}|{hard_case.get('fen')}|{variant_index}"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a candidate-choice regression eval set from hard cases."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--variants-per-case", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()

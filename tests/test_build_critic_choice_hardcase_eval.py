from __future__ import annotations

import json
from pathlib import Path

from finetune.build_critic_choice_hardcase_eval import (
    HardcaseEvalConfig,
    build_hardcase_eval_dataset,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_build_hardcase_eval_dataset_writes_choice_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "hard_cases.jsonl"
    output_path = tmp_path / "hardcase_eval.jsonl"
    input_path.write_text(json.dumps(_hard_case()) + "\n", encoding="utf-8")

    stats = build_hardcase_eval_dataset(
        HardcaseEvalConfig(
            input=str(input_path),
            output=str(output_path),
            metadata_output=None,
            variants_per_case=2,
            seed=7,
        )
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.input_hard_cases == 1
    assert stats.rows_written == 2
    assert stats.by_category["selected_blunder"] == 1
    assert stats.by_target_risk["good"] == 1
    assert rows[0]["fen"] == START_FEN
    assert rows[0]["move"] == "g1f3"
    assert rows[0]["completion"] == '{"move":"g1f3"}'
    assert rows[0]["split"] == "hardcase_eval"
    assert rows[0]["target_risk"] == "good"
    assert rows[0]["hard_case_selected_move"] == "f2f3"
    assert rows[0]["hard_case_categories"] == ["selected_blunder", "missed_good"]
    assert "Candidate moves:" in rows[0]["prompt"]
    assert len(rows[0]["candidates"]) == 3
    assert rows[1]["hard_case_variant_index"] == 1


def _hard_case() -> dict[str, object]:
    return {
        "example_index": 4,
        "categories": ["selected_blunder", "missed_good"],
        "fen": START_FEN,
        "side_to_move": "white",
        "predicted_move": "f2f3",
        "target": _candidate("g1f3", "Nf3", "stockfish_good", "good", 100, 0),
        "cpl_regret_vs_oracle": 500,
        "cpl_delta_vs_first": 420,
        "candidates": [
            _candidate("e2e4", "e4", "arena_move", "playable", 80, 80),
            _candidate("g1f3", "Nf3", "stockfish_good", "good", 100, 0),
            _candidate("f2f3", "f3", "random_legal", "blunder", 0, 500),
        ],
    }


def _candidate(
    uci: str,
    san: str,
    source: str,
    risk: str,
    score: int,
    cpl: int,
) -> dict[str, object]:
    return {
        "uci": uci,
        "san": san,
        "source": source,
        "risk": risk,
        "score": score,
        "centipawn_loss": cpl,
        "candidate_rank_in_generator": None,
    }

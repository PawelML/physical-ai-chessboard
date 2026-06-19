from __future__ import annotations

import json
from pathlib import Path

from finetune.analyze_pairwise_comparison_predictions import analyze_pairwise_predictions
from finetune.build_pairwise_comparison_eval import (
    PairwiseComparisonEvalConfig,
    build_pairwise_comparison_eval,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_pairwise_comparison_eval_expands_choice_rows(tmp_path: Path) -> None:
    choice_path = tmp_path / "choice.jsonl"
    pairwise_path = tmp_path / "pairwise.jsonl"
    _write_jsonl(choice_path, [_choice_row()])

    stats = build_pairwise_comparison_eval(
        PairwiseComparisonEvalConfig(
            input=str(choice_path),
            output=str(pairwise_path),
            metadata_output=None,
        )
    )

    rows = [json.loads(line) for line in pairwise_path.read_text(encoding="utf-8").splitlines()]

    assert stats.choice_rows_seen == 1
    assert stats.choice_rows_written == 1
    assert stats.pair_rows_written == 3
    assert {tuple(row["pair_moves"]) for row in rows} == {
        ("e2e4", "g1f3"),
        ("g1f3", "f2f3"),
        ("e2e4", "f2f3"),
    }
    assert rows[0]["pair_source_choice_index"] == 0
    assert rows[0]["completion"].startswith('{"move":"')
    assert "Choose the safer move from the two candidates." in rows[0]["prompt"]


def test_analyze_pairwise_comparison_predictions_composes_votes(tmp_path: Path) -> None:
    choice_path = tmp_path / "choice.jsonl"
    pairwise_path = tmp_path / "pairwise.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    _write_jsonl(choice_path, [_choice_row()])
    build_pairwise_comparison_eval(
        PairwiseComparisonEvalConfig(
            input=str(choice_path),
            output=str(pairwise_path),
            metadata_output=None,
        )
    )
    pair_rows = [
        json.loads(line) for line in pairwise_path.read_text(encoding="utf-8").splitlines()
    ]
    _write_jsonl(
        predictions_path,
        [_prediction(_preferred_pair_move(row["pair_moves"])) for row in pair_rows],
    )

    report = analyze_pairwise_predictions(
        choice_dataset_path=choice_path,
        pairwise_dataset_path=pairwise_path,
        predictions_path=predictions_path,
    )

    assert report["examples"] == 1
    assert report["pair_examples"] == 3
    assert report["pair_candidate_ok"] == 3
    assert report["top1_match"] == 1
    assert report["selected_mean_centipawn_loss"] == 0.0
    assert report["first_prompt_candidate_mean_centipawn_loss"] == 80.0
    assert report["selected_risk_counts"] == {"good": 1}
    assert report["selected_prompt_index"] == {2: 1}
    assert report["improved_vs_first_count"] == 1


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _choice_row() -> dict[str, object]:
    return {
        "prompt": "prompt",
        "completion": '{"move":"g1f3"}',
        "fen": START_FEN,
        "move": "g1f3",
        "fen_before": START_FEN,
        "side_to_move": "white",
        "split": "validation",
        "candidate_count": 3,
        "candidates": [
            _candidate("e2e4", "e4", "playable", 80, 80),
            _candidate("g1f3", "Nf3", "good", 100, 0),
            _candidate("f2f3", "f3", "blunder", 0, 500),
        ],
        "target_prompt_index": 2,
    }


def _candidate(move: str, san: str, risk: str, score: int, cpl: int) -> dict[str, object]:
    return {
        "uci": move,
        "san": san,
        "source": "arena_candidate",
        "risk": risk,
        "score": score,
        "centipawn_loss": cpl,
        "candidate_rank_in_generator": None,
    }


def _prediction(move: str) -> dict[str, object]:
    return {
        "parsed_move": move,
        "parse_ok": True,
        "legal_ok": True,
    }


def _preferred_pair_move(pair_moves: list[str]) -> str:
    if "g1f3" in pair_moves:
        return "g1f3"
    return "e2e4"

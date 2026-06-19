from __future__ import annotations

import json
from pathlib import Path

from finetune.analyze_critic_choice_predictions import analyze_predictions


def test_analyze_critic_choice_predictions_reports_candidate_cpl(tmp_path: Path) -> None:
    dataset = tmp_path / "choice.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        dataset,
        [
            _choice_row(
                target="g1f3",
                candidates=[
                    _candidate("e2e4", "playable", 80),
                    _candidate("g1f3", "good", 0),
                ],
                target_index=2,
            ),
            _choice_row(
                target="d2d4",
                candidates=[
                    _candidate("d2d4", "good", 0),
                    _candidate("f2f3", "blunder", 500),
                ],
                target_index=1,
            ),
        ],
    )
    _write_jsonl(
        predictions,
        [
            {"parsed_move": "g1f3", "parse_ok": True, "legal_ok": True},
            {"parsed_move": "f2f3", "parse_ok": True, "legal_ok": True},
        ],
    )

    report = analyze_predictions(dataset_path=dataset, predictions_path=predictions)

    assert report["examples"] == 2
    assert report["candidate_ok"] == 2
    assert report["top1_match"] == 1
    assert report["selected_mean_centipawn_loss"] == 250.0
    assert report["target_oracle_mean_centipawn_loss"] == 0.0
    assert report["first_prompt_candidate_mean_centipawn_loss"] == 40.0
    assert report["selected_risk_counts"] == {"blunder": 1, "good": 1}
    assert report["target_prompt_index"] == {1: 1, 2: 1}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _choice_row(
    *,
    target: str,
    candidates: list[dict[str, object]],
    target_index: int,
) -> dict[str, object]:
    return {
        "move": target,
        "target_prompt_index": target_index,
        "candidates": candidates,
    }


def _candidate(move: str, risk: str, cpl: int) -> dict[str, object]:
    return {
        "uci": move,
        "risk": risk,
        "centipawn_loss": cpl,
    }

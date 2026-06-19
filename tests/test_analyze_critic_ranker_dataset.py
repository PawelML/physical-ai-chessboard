from __future__ import annotations

import json
from pathlib import Path

from finetune.analyze_critic_ranker_dataset import analyze_dataset


def test_analyze_critic_ranker_dataset_reports_ranking_signal(tmp_path: Path) -> None:
    dataset = tmp_path / "critic.jsonl"
    rows = [
        _row("fen-a", "e2e4", "arena_move", "blunder", 0, 500, None),
        _row("fen-a", "g1f3", "arena_candidate", "good", 100, 0, 0),
        _row("fen-a", "d2d4", "arena_candidate", "playable", 80, 80, 1),
        _row("fen-b", "e7e5", "arena_move", "playable", 75, 100, None),
        _row("fen-b", "b8c6", "random_legal", "mistake", 40, 240, None),
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = analyze_dataset(dataset)

    assert report["rows"] == 5
    assert report["positions"] == 2
    assert report["mixed_safe_unsafe_positions"] == 2
    assert report["positions_with_qwen_candidates"] == 1
    assert report["risk_counts"] == {"blunder": 1, "good": 1, "mistake": 1, "playable": 2}
    assert report["first_generator_candidate"]["risk_counts"] == {"good": 1}
    assert report["oracle_gain_vs_arena_final"] == {
        "positions": 2,
        "mean_score_gain": 50.0,
        "mean_centipawn_loss_reduction": 250.0,
        "improved_positions_by_score": 1,
        "improved_positions_by_cpl": 1,
    }
    assert report["final_blunder_oracle_safe"] == 1
    assert report["first_generator_blunder_oracle_safe"] == 0


def _row(
    fen: str,
    move: str,
    source: str,
    risk: str,
    score: int,
    cpl: int,
    rank: int | None,
) -> dict[str, object]:
    return {
        "fen_before": fen,
        "candidate_uci": move,
        "source": source,
        "risk": risk,
        "score": score,
        "centipawn_loss": cpl,
        "candidate_rank_in_generator": rank,
    }

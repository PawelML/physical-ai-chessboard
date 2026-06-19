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
        _row("fen-c", "c2c4", "policy_sample", "blunder", 0, 420, 1),
        _row("fen-c", "g1f3", "policy_sample", "good", 95, 20, 0),
        _row("fen-d", "b1c3", "arena_candidate_pairwise", "mistake", 40, 220, 0),
        _row("fen-d", "g1f3", "arena_candidate_pairwise", "good", 100, 0, 1),
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = analyze_dataset(dataset)

    assert report["rows"] == 9
    assert report["positions"] == 4
    assert report["mixed_safe_unsafe_positions"] == 4
    assert report["positions_with_qwen_candidates"] == 3
    assert report["risk_counts"] == {"blunder": 2, "good": 3, "mistake": 2, "playable": 2}
    assert report["first_generator_candidate"]["risk_counts"] == {"good": 2, "mistake": 1}
    assert report["oracle_gain_vs_arena_final"] == {
        "positions": 2,
        "mean_score_gain": 50.0,
        "mean_centipawn_loss_reduction": 250.0,
        "improved_positions_by_score": 1,
        "improved_positions_by_cpl": 1,
    }
    assert report["final_blunder_oracle_safe"] == 1
    assert report["first_generator_blunder_oracle_safe"] == 0
    assert report["oracle_gain_vs_first_generator"] == {
        "positions": 3,
        "mean_score_gain": 20.0,
        "mean_centipawn_loss_reduction": 73.33,
        "improved_positions_by_score": 1,
        "improved_positions_by_cpl": 1,
    }


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

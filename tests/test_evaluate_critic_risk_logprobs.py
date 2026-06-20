from __future__ import annotations

import pytest

from finetune.evaluate_critic_risk_logprobs import ranking_metrics, softmax


def test_softmax_normalizes_logprob_scores() -> None:
    probabilities = softmax({"good": 0.0, "playable": -1.0, "mistake": -2.0})

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["good"] > probabilities["playable"] > probabilities["mistake"]


def test_ranking_metrics_selects_by_expected_risk_score() -> None:
    predictions = [
        _prediction("fen-a", "e2e4", 300, 1, "blunder", 0, 0.1),
        _prediction("fen-a", "g1f3", 20, 2, "good", 95, 2.8),
        _prediction("fen-b", "d2d4", 40, 1, "good", 90, 1.7),
        _prediction("fen-b", "c2c4", 170, 2, "mistake", 50, 2.3),
    ]

    metrics = ranking_metrics(predictions)

    assert metrics == {
        "positions": 2,
        "selected_mean_centipawn_loss": 95.0,
        "oracle_mean_centipawn_loss": 30.0,
        "first_generator_mean_centipawn_loss": 170.0,
        "selected_blunders": 0,
        "oracle_blunders": 0,
        "first_generator_blunders": 1,
        "oracle_match_by_move": 1,
        "improved_vs_first_count": 1,
        "worsened_vs_first_count": 1,
        "tied_first_count": 0,
    }


def _prediction(
    fen: str,
    move: str,
    cpl: int,
    generator_rank: int,
    target_risk: str,
    target_score: int,
    expected_score: float,
) -> dict[str, object]:
    return {
        "fen_before": fen,
        "candidate_uci": move,
        "candidate_rank_in_generator": generator_rank,
        "target": {
            "risk": target_risk,
            "blunder": target_risk == "blunder",
            "score": target_score,
            "reason_type": "none",
        },
        "centipawn_loss": cpl,
        "expected_risk_score": expected_score,
    }

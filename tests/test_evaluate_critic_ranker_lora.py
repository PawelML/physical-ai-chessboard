from __future__ import annotations

from finetune.evaluate_critic_ranker_lora import _ranking_metrics, parse_critic_json


def test_parse_critic_json_extracts_valid_object_from_model_text() -> None:
    parsed = parse_critic_json(
        '<|im_start|>assistant\n{"risk":"good","blunder":false,'
        '"score":98,"reason_type":"none"}<|im_end|>'
    )

    assert parsed == {
        "risk": "good",
        "blunder": False,
        "score": 98,
        "reason_type": "none",
    }


def test_parse_critic_json_rejects_invalid_schema() -> None:
    assert parse_critic_json('{"risk":"safe","blunder":false,"score":98}') is None
    assert parse_critic_json('{"risk":"good","blunder":"false","score":98}') is None


def test_ranking_metrics_selects_predicted_best_candidate() -> None:
    predictions = [
        _prediction("fen-a", "e2e4", 500, "blunder", 0, "blunder", 10),
        _prediction("fen-a", "g1f3", 0, "good", 100, "good", 90),
        _prediction("fen-b", "d2d4", 80, "playable", 80, "good", 95),
        _prediction("fen-b", "c2c4", 30, "good", 92, "playable", 70),
    ]

    metrics = _ranking_metrics(predictions)

    assert metrics == {
        "positions": 2,
        "selected_mean_centipawn_loss": 40.0,
        "oracle_mean_centipawn_loss": 15.0,
        "selected_blunders": 0,
        "oracle_blunders": 0,
        "oracle_match_by_move": 1,
    }


def _prediction(
    fen: str,
    move: str,
    cpl: int,
    target_risk: str,
    target_score: int,
    predicted_risk: str,
    predicted_score: int,
) -> dict[str, object]:
    return {
        "fen_before": fen,
        "candidate_uci": move,
        "target": {
            "risk": target_risk,
            "blunder": target_risk == "blunder",
            "score": target_score,
            "reason_type": "none",
        },
        "centipawn_loss": cpl,
        "predicted": {
            "risk": predicted_risk,
            "blunder": predicted_risk == "blunder",
            "score": predicted_score,
            "reason_type": "none",
        },
    }

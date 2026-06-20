from __future__ import annotations

import chess
import pytest

from arena_core.risk_logprob_scorer import build_risk_prompt, softmax


def test_build_risk_prompt_matches_critic_ranker_shape() -> None:
    board = chess.Board()

    prompt = build_risk_prompt(board=board, move=chess.Move.from_uci("e2e4"))

    assert "You are a chess move risk classifier." in prompt
    assert f"FEN before: {chess.Board().fen()}" in prompt
    assert "Side to move: white" in prompt
    assert "Candidate move: e2e4 (e4)" in prompt
    assert "FEN after candidate:" in prompt
    assert "Opponent legal replies:" in prompt
    assert '{"risk":"blunder|mistake|playable|good","blunder":true,' in prompt


def test_softmax_prefers_highest_logprob() -> None:
    probabilities = softmax({"blunder": -3.0, "mistake": -2.0, "good": 0.0})

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities["good"] > probabilities["mistake"] > probabilities["blunder"]

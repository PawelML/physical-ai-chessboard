import chess

from arena_core.prompts import build_strict_prompt


def test_constrained_prompt_uses_uci_only_move_contract() -> None:
    prompt = build_strict_prompt(
        board=chess.Board(),
        san_history=[],
        own_moves=[],
        last_opponent_move=None,
        legality_mode="constrained",
    )

    assert '{"move":"e2e4"}' in prompt.text
    assert "The move value must be UCI coordinate notation" in prompt.text
    assert "Legal moves (UCI):" in prompt.text
    assert "e2e4" in prompt.text
    assert "Legal moves (SAN):" not in prompt.text

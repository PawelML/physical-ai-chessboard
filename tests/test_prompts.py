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


def test_prompt_template_hash_tracks_template_not_position() -> None:
    board_after_e4 = chess.Board()
    board_after_e4.push_uci("e2e4")
    start_prompt = build_strict_prompt(
        board=chess.Board(),
        san_history=[],
        own_moves=[],
        last_opponent_move=None,
        legality_mode="open",
    )
    later_prompt = build_strict_prompt(
        board=board_after_e4,
        san_history=["e4"],
        own_moves=[],
        last_opponent_move="e4/e2e4",
        legality_mode="open",
    )
    constrained_prompt = build_strict_prompt(
        board=chess.Board(),
        san_history=[],
        own_moves=[],
        last_opponent_move=None,
        legality_mode="constrained",
    )

    assert start_prompt.text != later_prompt.text
    assert start_prompt.template_hash == later_prompt.template_hash
    assert start_prompt.template_hash != constrained_prompt.template_hash

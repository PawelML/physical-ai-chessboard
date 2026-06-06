from arena_core.parser import parse_uci_json


def test_parse_uci_json_accepts_move() -> None:
    parsed = parse_uci_json('{"move":"e2e4"}')

    assert parsed.parse_ok is True
    assert parsed.move == "e2e4"
    assert parsed.error_type is None


def test_parse_uci_json_rejects_non_json() -> None:
    parsed = parse_uci_json("e2e4")

    assert parsed.parse_ok is False
    assert parsed.error_type == "malformed_json"

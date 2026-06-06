from arena_core.parser import parse_uci_json


def test_parse_uci_json_accepts_move() -> None:
    parsed = parse_uci_json('{"move":"e2e4"}')

    assert parsed.parse_ok is True
    assert parsed.move == "e2e4"
    assert parsed.error_type is None


def test_parse_uci_json_accepts_fenced_json() -> None:
    parsed = parse_uci_json('```json\n{"move":"e2e4"}\n```')

    assert parsed.parse_ok is True
    assert parsed.move == "e2e4"


def test_parse_uci_json_accepts_prose_then_json() -> None:
    parsed = parse_uci_json('I choose this move:\n{"move":"g1f3"}\nGood luck.')

    assert parsed.parse_ok is True
    assert parsed.move == "g1f3"


def test_parse_uci_json_rejects_non_json() -> None:
    parsed = parse_uci_json("e2e4")

    assert parsed.parse_ok is False
    assert parsed.error_type == "malformed_json"


def test_parse_uci_json_rejects_genuinely_malformed() -> None:
    parsed = parse_uci_json('I choose {"move":')

    assert parsed.parse_ok is False
    assert parsed.error_type == "malformed_json"

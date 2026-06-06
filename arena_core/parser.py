import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedMove:
    move: str | None
    parse_ok: bool
    error_type: str | None
    reason: str | None


def parse_uci_json(raw_response: str) -> ParsedMove:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return ParsedMove(None, False, "malformed_json", f"Response is not JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return ParsedMove(None, False, "malformed_json", "Response JSON must be an object")
    move = payload.get("move")
    if not isinstance(move, str) or not move.strip():
        return ParsedMove(None, False, "malformed_json", "Response must contain a non-empty move")
    return ParsedMove(move.strip(), True, None, None)

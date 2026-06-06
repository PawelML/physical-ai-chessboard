import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedMove:
    move: str | None
    parse_ok: bool
    error_type: str | None
    reason: str | None


def parse_uci_json(raw_response: str) -> ParsedMove:
    normalized = _strip_code_fence(raw_response.strip())
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        object_text = _first_balanced_object(normalized)
        if object_text is None:
            return ParsedMove(None, False, "malformed_json", "Response does not contain JSON")
        try:
            payload = json.loads(object_text)
        except json.JSONDecodeError as exc:
            return ParsedMove(None, False, "malformed_json", f"Response is not JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return ParsedMove(None, False, "malformed_json", "Response JSON must be an object")
    move = payload.get("move")
    if not isinstance(move, str) or not move.strip():
        return ParsedMove(None, False, "malformed_json", "Response must contain a non-empty move")
    return ParsedMove(move.strip(), True, None, None)


def _strip_code_fence(value: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def _first_balanced_object(value: str) -> str | None:
    start = value.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(value[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return None

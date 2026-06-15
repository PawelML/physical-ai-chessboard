from hashlib import sha256

import chess


def close_if_present(item: object) -> None:
    close = getattr(item, "close", None)
    if callable(close):
        close()


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def own_moves_for_side(
    *,
    san_history: list[str],
    uci_history: list[str],
    side: chess.Color,
) -> list[tuple[str, str]]:
    start = 0 if side == chess.WHITE else 1
    return list(zip(san_history[start::2], uci_history[start::2], strict=True))


def last_opponent_move(
    *,
    san_history: list[str],
    uci_history: list[str],
    side_to_move: chess.Color,
) -> str | None:
    if not san_history:
        return None
    last_move_was_white = len(san_history) % 2 == 1
    opponent_just_moved = (
        side_to_move == chess.BLACK and last_move_was_white
    ) or (side_to_move == chess.WHITE and not last_move_was_white)
    if not opponent_just_moved:
        return None
    return f"{san_history[-1]}/{uci_history[-1]}"

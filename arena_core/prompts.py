from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import chess

PromptMode = Literal["strict", "reasoning"]
LegalityMode = Literal["open", "constrained"]

STRICT_TEMPLATE_VERSION = "strict-v1"


@dataclass(frozen=True)
class BuiltPrompt:
    version: str
    mode: PromptMode
    legality_mode: LegalityMode
    text: str
    template_hash: str


def template_hash(version: str, mode: PromptMode, legality_mode: LegalityMode) -> str:
    return sha256(f"{version}:{mode}:{legality_mode}".encode()).hexdigest()


def build_strict_prompt(
    *,
    board: chess.Board,
    san_history: list[str],
    own_moves: list[tuple[str, str]],
    last_opponent_move: str | None,
    legality_mode: LegalityMode,
    feedback: dict[str, object] | None = None,
    include_ascii_board: bool = True,
    version: str = STRICT_TEMPLATE_VERSION,
) -> BuiltPrompt:
    side = "white" if board.turn == chess.WHITE else "black"
    parts = [
        "You are playing a benchmark chess game.",
        "Return only strict JSON in this exact shape: {\"move\":\"e2e4\"}.",
        f"Side to move: {side}.",
        f"Current FEN: {board.fen()}",
        f"SAN history: {' '.join(san_history) if san_history else '(start position)'}",
        "Your prior moves: "
        + (
            ", ".join(f"{san}/{uci}" for san, uci in own_moves)
            if own_moves
            else "(none)"
        ),
        f"Last opponent move: {last_opponent_move or '(none)'}",
    ]
    if include_ascii_board:
        parts.append(f"ASCII board:\n{board}")
    if legality_mode == "constrained" or feedback is not None:
        legal_moves = sorted(move.uci() for move in board.legal_moves)
        parts.append("Legal moves: " + ", ".join(legal_moves))
    if feedback is not None:
        parts.append(f"Previous attempt feedback: {feedback}")
    text = "\n".join(parts)
    return BuiltPrompt(
        version=version,
        mode="strict",
        legality_mode=legality_mode,
        text=text,
        template_hash=template_hash(version, "strict", legality_mode),
    )


def build_reasoning_prompt(
    *,
    fen_before: str,
    accepted_san: str,
    accepted_uci: str,
    persona: str,
    evaluation_label: str,
    legality_mode: LegalityMode,
    version: str = "reasoning-v1",
) -> BuiltPrompt:
    text = "\n".join(
        [
            "Annotate an already-scored strict chess move.",
            "Do not propose a different move.",
            f"Persona: {persona}.",
            f"FEN before move: {fen_before}",
            f"Accepted move: {accepted_san} / {accepted_uci}",
            f"Engine classification: {evaluation_label}",
            "Return concise commentary for a match report.",
        ]
    )
    return BuiltPrompt(
        version=version,
        mode="reasoning",
        legality_mode=legality_mode,
        text=text,
        template_hash=template_hash(version, "reasoning", legality_mode),
    )

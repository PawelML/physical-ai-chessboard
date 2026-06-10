from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import chess

PromptMode = Literal["strict", "reasoning"]
LegalityMode = Literal["open", "constrained"]

STRICT_TEMPLATE_VERSION = "strict-v7"


@dataclass(frozen=True)
class BuiltPrompt:
    version: str
    mode: PromptMode
    legality_mode: LegalityMode
    text: str
    template_hash: str


def template_hash(template_text: str) -> str:
    return sha256(template_text.encode("utf-8")).hexdigest()


def build_strict_prompt(
    *,
    board: chess.Board,
    san_history: list[str],
    own_moves: list[tuple[str, str]],
    last_opponent_move: str | None,
    legality_mode: LegalityMode,
    feedback: dict[str, object] | None = None,
    strategic_memory: dict[str, str] | None = None,
    repetition_warning: str | None = None,
    include_ascii_board: bool = True,
    version: str = STRICT_TEMPLATE_VERSION,
) -> BuiltPrompt:
    side = "white" if board.turn == chess.WHITE else "black"
    parts = [
        "You are playing a competitive chess game and your goal is to win.",
        (
            "Win by checkmating the opponent's king; if a win is out of reach, "
            "play for a draw rather than a loss."
        ),
        (
            "Among the legal moves, choose the strongest one: develop your pieces, "
            "fight for the center, keep your own king safe, and never give away "
            "material for free."
        ),
        (
            "Before committing, look for checks, captures and threats — both your own "
            "tactical chances and the opponent's threats against you."
        ),
        (
            "Return only strict JSON in this exact shape: "
            "{\"move\":\"e2e4\",\"rationale\":\"short reason\","
            "\"strategy_update\":{\"objective\":\"short updated plan\"}}."
            if strategic_memory is not None
            else "Return only strict JSON in this exact shape: {\"move\":\"e2e4\"}."
        ),
        (
            "The move value must be UCI coordinate notation, e.g. e2e4, g1f3, "
            "e1g1, e7e8q. Do not use SAN moves like e4, Nf3, O-O, or Qxc4."
        ),
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
    if strategic_memory is not None:
        parts.extend(
            [
                "Your private strategic memory:",
                f"- Objective: {strategic_memory.get('objective', '(none)')}",
                f"- Opponent threats: {strategic_memory.get('opponent_threats', '(none)')}",
                f"- Pieces to improve: {strategic_memory.get('pieces_to_improve', '(none)')}",
                f"- Avoid: {strategic_memory.get('avoid', '(none)')}",
                f"- Last own move rationale: {strategic_memory.get('last_rationale', '(none)')}",
                "Update this memory after choosing a move. Keep each field short and concrete.",
            ]
        )
    if repetition_warning is not None:
        parts.append(f"Repetition warning: {repetition_warning}")
    if legality_mode == "constrained" or feedback is not None:
        legal_moves = sorted(move.uci() for move in board.legal_moves)
        parts.extend(
            [
                "Choose exactly one move from this legal move list.",
                "Your move must exactly match one listed UCI move.",
                "If strategic memory conflicts with the legal move list, ignore the memory.",
                "Legal moves (UCI): " + ", ".join(legal_moves),
            ]
        )
    if feedback is not None:
        parts.extend(
            [
                "Your previous response was invalid.",
                f"Do not repeat this attempted move: {feedback.get('attempted_move') or '(none)'}.",
                "Choose a different move that exactly appears in the legal move list.",
            ]
        )
        parts.append(f"Previous attempt feedback: {feedback}")
    text = "\n".join(parts)
    skeleton = _strict_template_skeleton(
        legality_mode=legality_mode,
        has_feedback=feedback is not None,
        has_strategic_memory=strategic_memory is not None,
        has_repetition_warning=repetition_warning is not None,
        include_ascii_board=include_ascii_board,
    )
    return BuiltPrompt(
        version=version,
        mode="strict",
        legality_mode=legality_mode,
        text=text,
        template_hash=template_hash(skeleton),
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
    parts = [
        "Annotate an already-scored strict chess move.",
        "Do not propose a different move.",
        f"Persona: {persona}.",
        f"FEN before move: {fen_before}",
        f"Accepted move: {accepted_san} / {accepted_uci}",
        f"Engine classification: {evaluation_label}",
        "Return concise commentary for a match report.",
    ]
    text = "\n".join(parts)
    return BuiltPrompt(
        version=version,
        mode="reasoning",
        legality_mode=legality_mode,
        text=text,
        template_hash=template_hash(
            "\n".join(
                [
                    "Annotate an already-scored strict chess move.",
                    "Do not propose a different move.",
                    "Persona: {persona}.",
                    "FEN before move: {fen_before}",
                    "Accepted move: {accepted_san} / {accepted_uci}",
                    "Engine classification: {evaluation_label}",
                    "Return concise commentary for a match report.",
                ]
            )
        ),
    )


def _strict_template_skeleton(
    *,
    legality_mode: LegalityMode,
    has_feedback: bool,
    has_strategic_memory: bool,
    has_repetition_warning: bool,
    include_ascii_board: bool,
) -> str:
    parts = [
        "You are playing a competitive chess game and your goal is to win.",
        (
            "Win by checkmating the opponent's king; if a win is out of reach, "
            "play for a draw rather than a loss."
        ),
        (
            "Among the legal moves, choose the strongest one: develop your pieces, "
            "fight for the center, keep your own king safe, and never give away "
            "material for free."
        ),
        (
            "Before committing, look for checks, captures and threats — both your own "
            "tactical chances and the opponent's threats against you."
        ),
        (
            "Return only strict JSON in this exact shape: "
            "{\"move\":\"e2e4\",\"rationale\":\"short reason\","
            "\"strategy_update\":{\"objective\":\"short updated plan\"}}."
            if has_strategic_memory
            else "Return only strict JSON in this exact shape: {\"move\":\"e2e4\"}."
        ),
        (
            "The move value must be UCI coordinate notation, e.g. e2e4, g1f3, "
            "e1g1, e7e8q. Do not use SAN moves like e4, Nf3, O-O, or Qxc4."
        ),
        "Side to move: {side}.",
        "Current FEN: {fen}",
        "SAN history: {san_history}",
        "Your prior moves: {own_moves}",
        "Last opponent move: {last_opponent_move}",
    ]
    if include_ascii_board:
        parts.append("ASCII board:\n{ascii_board}")
    if has_strategic_memory:
        parts.extend(
            [
                "Your private strategic memory:",
                "- Objective: {objective}",
                "- Opponent threats: {opponent_threats}",
                "- Pieces to improve: {pieces_to_improve}",
                "- Avoid: {avoid}",
                "- Last own move rationale: {last_rationale}",
                "Update this memory after choosing a move. Keep each field short and concrete.",
            ]
        )
    if has_repetition_warning:
        parts.append("Repetition warning: {repetition_warning}")
    if legality_mode == "constrained" or has_feedback:
        parts.extend(
            [
                "Choose exactly one move from this legal move list.",
                "Your move must exactly match one listed UCI move.",
                "If strategic memory conflicts with the legal move list, ignore the memory.",
                "Legal moves (UCI): {legal_moves}",
            ]
        )
    if has_feedback:
        parts.extend(
            [
                "Your previous response was invalid.",
                "Do not repeat this attempted move: {attempted_move}.",
                "Choose a different move that exactly appears in the legal move list.",
                "Previous attempt feedback: {feedback}",
            ]
        )
    return "\n".join(parts)

from dataclasses import dataclass
from typing import Literal

import chess

from arena_core.utils import text_hash

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


@dataclass
class _PromptTemplate:
    rendered_parts: list[str]
    skeleton_parts: list[str]

    def add(self, template: str, **values: object) -> None:
        self.rendered_parts.append(template.format(**values) if values else template)
        self.skeleton_parts.append(template)

    def extend(self, templates: list[str]) -> None:
        self.rendered_parts.extend(templates)
        self.skeleton_parts.extend(templates)

    def rendered(self) -> str:
        return "\n".join(self.rendered_parts)

    def skeleton(self) -> str:
        return "\n".join(self.skeleton_parts)


def template_hash(template_text: str) -> str:
    return text_hash(template_text)


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
    template = _PromptTemplate(rendered_parts=[], skeleton_parts=[])
    template.extend(
        [
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
        ]
    )
    template.add("Side to move: {side}.", side=side)
    template.add("Current FEN: {fen}", fen=board.fen())
    template.add(
        "SAN history: {san_history}",
        san_history=" ".join(san_history) if san_history else "(start position)",
    )
    template.add(
        "Your prior moves: {own_moves}",
        own_moves=(
            ", ".join(f"{san}/{uci}" for san, uci in own_moves) if own_moves else "(none)"
        ),
    )
    template.add(
        "Last opponent move: {last_opponent_move}",
        last_opponent_move=last_opponent_move or "(none)",
    )
    if include_ascii_board:
        template.add("ASCII board:\n{ascii_board}", ascii_board=board)
    if strategic_memory is not None:
        template.add("Your private strategic memory:")
        template.add(
            "- Objective: {objective}",
            objective=strategic_memory.get("objective", "(none)"),
        )
        template.add(
            "- Opponent threats: {opponent_threats}",
            opponent_threats=strategic_memory.get("opponent_threats", "(none)"),
        )
        template.add(
            "- Pieces to improve: {pieces_to_improve}",
            pieces_to_improve=strategic_memory.get("pieces_to_improve", "(none)"),
        )
        template.add("- Avoid: {avoid}", avoid=strategic_memory.get("avoid", "(none)"))
        template.add(
            "- Last own move rationale: {last_rationale}",
            last_rationale=strategic_memory.get("last_rationale", "(none)"),
        )
        template.add(
            "Update this memory after choosing a move. Keep each field short and concrete."
        )
    if repetition_warning is not None:
        template.add(
            "Repetition warning: {repetition_warning}",
            repetition_warning=repetition_warning,
        )
    if legality_mode == "constrained" or feedback is not None:
        legal_moves = sorted(move.uci() for move in board.legal_moves)
        template.extend(
            [
                "Choose exactly one move from this legal move list.",
                "Your move must exactly match one listed UCI move.",
                "If strategic memory conflicts with the legal move list, ignore the memory.",
            ]
        )
        template.add("Legal moves (UCI): {legal_moves}", legal_moves=", ".join(legal_moves))
    if feedback is not None:
        template.add("Your previous response was invalid.")
        template.add(
            "Do not repeat this attempted move: {attempted_move}.",
            attempted_move=feedback.get("attempted_move") or "(none)",
        )
        template.add("Choose a different move that exactly appears in the legal move list.")
        template.add("Previous attempt feedback: {feedback}", feedback=feedback)
    return BuiltPrompt(
        version=version,
        mode="strict",
        legality_mode=legality_mode,
        text=template.rendered(),
        template_hash=template_hash(template.skeleton()),
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
    template = _PromptTemplate(rendered_parts=[], skeleton_parts=[])
    template.extend(
        [
            "Annotate an already-scored strict chess move.",
            "Do not propose a different move.",
        ]
    )
    template.add("Persona: {persona}.", persona=persona)
    template.add("FEN before move: {fen_before}", fen_before=fen_before)
    template.add(
        "Accepted move: {accepted_san} / {accepted_uci}",
        accepted_san=accepted_san,
        accepted_uci=accepted_uci,
    )
    template.add("Engine classification: {evaluation_label}", evaluation_label=evaluation_label)
    template.add("Return concise commentary for a match report.")
    return BuiltPrompt(
        version=version,
        mode="reasoning",
        legality_mode=legality_mode,
        text=template.rendered(),
        template_hash=template_hash(template.skeleton()),
    )

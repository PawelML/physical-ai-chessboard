from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn

from arena_core.prompts import STRICT_TEMPLATE_VERSION, LegalityMode, build_strict_prompt


@dataclass(frozen=True)
class SmokeDatasetConfig:
    pgn: str
    output: str
    max_examples: int
    seed: int
    constrained_ratio: float
    include_ascii_board: bool
    prompt_version: str


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    config = SmokeDatasetConfig(
        pgn=str(args.pgn),
        output=str(args.output),
        max_examples=args.max_examples,
        seed=args.seed,
        constrained_ratio=args.constrained_ratio,
        include_ascii_board=not args.no_ascii_board,
        prompt_version=STRICT_TEMPLATE_VERSION,
    )
    count = build_dataset(
        pgn_path=args.pgn,
        output_path=args.output,
        max_examples=args.max_examples,
        constrained_ratio=args.constrained_ratio,
        include_ascii_board=not args.no_ascii_board,
        rng=rng,
    )
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "examples": count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {count} examples to {args.output}")


def build_dataset(
    *,
    pgn_path: Path,
    output_path: Path,
    max_examples: int,
    constrained_ratio: float,
    include_ascii_board: bool,
    rng: random.Random,
) -> int:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if not 0 <= constrained_ratio <= 1:
        raise ValueError("constrained_ratio must be between 0 and 1")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with pgn_path.open(encoding="utf-8", errors="replace") as pgn_file:
        with output_path.open("w", encoding="utf-8") as output_file:
            while count < max_examples:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                count += _write_game_examples(
                    game=game,
                    output_file=output_file,
                    remaining=max_examples - count,
                    constrained_ratio=constrained_ratio,
                    include_ascii_board=include_ascii_board,
                    rng=rng,
                )
    return count


def _write_game_examples(
    *,
    game: chess.pgn.Game,
    output_file: TextIO,
    remaining: int,
    constrained_ratio: float,
    include_ascii_board: bool,
    rng: random.Random,
) -> int:
    board = game.board()
    san_history: list[str] = []
    uci_history: list[str] = []
    count = 0
    for move in game.mainline_moves():
        if count >= remaining:
            break
        if move not in board.legal_moves:
            break

        mover = board.turn
        san = board.san(move)
        legality_mode: LegalityMode = (
            "constrained" if rng.random() < constrained_ratio else "open"
        )
        prompt = build_strict_prompt(
            board=board,
            san_history=san_history,
            own_moves=_own_moves_for_side(
                san_history=san_history,
                uci_history=uci_history,
                side=mover,
            ),
            last_opponent_move=_last_opponent_move(
                san_history=san_history,
                uci_history=uci_history,
                side_to_move=mover,
            ),
            legality_mode=legality_mode,
            include_ascii_board=include_ascii_board,
        )
        output_file.write(
            json.dumps(
                {
                    "prompt": prompt.text,
                    "completion": f'{{"move":"{move.uci()}"}}',
                    "move": move.uci(),
                    "san": san,
                    "fen": board.fen(),
                    "prompt_version": prompt.version,
                    "prompt_template_hash": prompt.template_hash,
                    "legality_mode": legality_mode,
                    "source": {
                        "event": game.headers.get("Event", "?"),
                        "site": game.headers.get("Site", "?"),
                        "date": game.headers.get("Date", "?"),
                        "round": game.headers.get("Round", "?"),
                        "white": game.headers.get("White", "?"),
                        "black": game.headers.get("Black", "?"),
                    },
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        count += 1
        board.push(move)
        san_history.append(san)
        uci_history.append(move.uci())
    return count


def _own_moves_for_side(
    *,
    san_history: list[str],
    uci_history: list[str],
    side: chess.Color,
) -> list[tuple[str, str]]:
    start = 0 if side == chess.WHITE else 1
    return list(zip(san_history[start::2], uci_history[start::2], strict=True))


def _last_opponent_move(
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small strict-v7 JSONL dataset from a PGN file."
    )
    parser.add_argument("--pgn", type=Path, required=True, help="Input PGN file.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path, usually under data/finetune/.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional sidecar JSON metadata path.",
    )
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constrained-ratio", type=float, default=0.3)
    parser.add_argument("--no-ascii-board", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

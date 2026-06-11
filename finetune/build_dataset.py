from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn

from arena_core.prompts import STRICT_TEMPLATE_VERSION, LegalityMode, build_strict_prompt

BLITZ_OR_SLOWER_SECONDS = 180
ESTIMATED_GAME_INCREMENT_MOVES = 40


@dataclass(frozen=True)
class DatasetConfig:
    pgn: str
    train_output: str
    val_output: str
    max_examples: int
    val_ratio: float
    seed: int
    constrained_ratio: float
    include_ascii_board: bool
    min_elo: int
    min_plies: int
    max_positions_per_game: int
    prompt_version: str = STRICT_TEMPLATE_VERSION


@dataclass
class BuildStats:
    games_seen: int = 0
    games_kept: int = 0
    examples_train: int = 0
    examples_val: int = 0
    skipped_games: Counter[str] = field(default_factory=Counter)

    @property
    def examples_total(self) -> int:
        return self.examples_train + self.examples_val

    def to_json(self) -> dict[str, object]:
        return {
            "games_seen": self.games_seen,
            "games_kept": self.games_kept,
            "examples_train": self.examples_train,
            "examples_val": self.examples_val,
            "examples_total": self.examples_total,
            "skipped_games": dict(sorted(self.skipped_games.items())),
        }


def main() -> None:
    args = _parse_args()
    config = DatasetConfig(
        pgn=args.pgn,
        train_output=str(args.train_output),
        val_output=str(args.val_output),
        max_examples=args.max_examples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        constrained_ratio=args.constrained_ratio,
        include_ascii_board=not args.no_ascii_board,
        min_elo=args.min_elo,
        min_plies=args.min_plies,
        max_positions_per_game=args.max_positions_per_game,
    )
    rng = random.Random(args.seed)
    with _open_pgn(args.pgn) as pgn_file:
        stats = build_dataset_from_stream(
            pgn_file=pgn_file,
            train_output_path=args.train_output,
            val_output_path=args.val_output,
            max_examples=args.max_examples,
            val_ratio=args.val_ratio,
            constrained_ratio=args.constrained_ratio,
            include_ascii_board=not args.no_ascii_board,
            min_elo=args.min_elo,
            min_plies=args.min_plies,
            max_positions_per_game=args.max_positions_per_game,
            rng=rng,
        )
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "stats": stats.to_json(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        "Wrote "
        f"{stats.examples_train} train and {stats.examples_val} val examples "
        f"from {stats.games_kept}/{stats.games_seen} kept games."
    )


def build_dataset(
    *,
    pgn_path: Path,
    train_output_path: Path,
    val_output_path: Path,
    max_examples: int,
    val_ratio: float,
    constrained_ratio: float,
    include_ascii_board: bool,
    min_elo: int,
    min_plies: int,
    max_positions_per_game: int,
    rng: random.Random,
) -> BuildStats:
    with pgn_path.open(encoding="utf-8", errors="replace") as pgn_file:
        return build_dataset_from_stream(
            pgn_file=pgn_file,
            train_output_path=train_output_path,
            val_output_path=val_output_path,
            max_examples=max_examples,
            val_ratio=val_ratio,
            constrained_ratio=constrained_ratio,
            include_ascii_board=include_ascii_board,
            min_elo=min_elo,
            min_plies=min_plies,
            max_positions_per_game=max_positions_per_game,
            rng=rng,
        )


def build_dataset_from_stream(
    *,
    pgn_file: TextIO,
    train_output_path: Path,
    val_output_path: Path,
    max_examples: int,
    val_ratio: float,
    constrained_ratio: float,
    include_ascii_board: bool,
    min_elo: int,
    min_plies: int,
    max_positions_per_game: int,
    rng: random.Random,
) -> BuildStats:
    _validate_args(
        max_examples=max_examples,
        val_ratio=val_ratio,
        constrained_ratio=constrained_ratio,
        min_elo=min_elo,
        min_plies=min_plies,
        max_positions_per_game=max_positions_per_game,
    )
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()
    with train_output_path.open("w", encoding="utf-8") as train_output:
        with val_output_path.open("w", encoding="utf-8") as val_output:
            while stats.examples_total < max_examples:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                stats.games_seen += 1
                moves = list(game.mainline_moves())
                skip_reason = _skip_reason(
                    game=game,
                    moves=moves,
                    min_elo=min_elo,
                    min_plies=min_plies,
                )
                if skip_reason is not None:
                    stats.skipped_games[skip_reason] += 1
                    continue

                split = "val" if rng.random() < val_ratio else "train"
                output_file = val_output if split == "val" else train_output
                remaining = max_examples - stats.examples_total
                written = _write_game_examples(
                    game=game,
                    moves=moves,
                    output_file=output_file,
                    split=split,
                    source_game_index=stats.games_seen,
                    remaining=remaining,
                    constrained_ratio=constrained_ratio,
                    include_ascii_board=include_ascii_board,
                    max_positions_per_game=max_positions_per_game,
                    rng=rng,
                )
                if written == 0:
                    stats.skipped_games["no_valid_sampled_positions"] += 1
                    continue

                stats.games_kept += 1
                if split == "val":
                    stats.examples_val += written
                else:
                    stats.examples_train += written
    return stats


def _write_game_examples(
    *,
    game: chess.pgn.Game,
    moves: list[chess.Move],
    output_file: TextIO,
    split: str,
    source_game_index: int,
    remaining: int,
    constrained_ratio: float,
    include_ascii_board: bool,
    max_positions_per_game: int,
    rng: random.Random,
) -> int:
    sample_indices = set(
        _evenly_spaced_indices(
            length=len(moves),
            limit=min(max_positions_per_game, remaining),
        )
    )
    board = game.board()
    san_history: list[str] = []
    uci_history: list[str] = []
    game_id = _game_id(game=game, moves=moves)
    count = 0

    for ply_index, move in enumerate(moves):
        if move not in board.legal_moves:
            break

        mover = board.turn
        san = board.san(move)
        if ply_index in sample_indices:
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
                        "split": split,
                        "game_id": game_id,
                        "source_game_index": source_game_index,
                        "ply": ply_index + 1,
                        "source": _source_metadata(game),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
            if count >= remaining:
                break

        board.push(move)
        san_history.append(san)
        uci_history.append(move.uci())
    return count


def _skip_reason(
    *,
    game: chess.pgn.Game,
    moves: list[chess.Move],
    min_elo: int,
    min_plies: int,
) -> str | None:
    headers = game.headers
    if not _is_rated(headers):
        return "unrated"
    if not _is_blitz_or_slower(headers):
        return "time_control_too_fast_or_unknown"
    if not _has_minimum_elos(headers, min_elo):
        return "elo_below_threshold_or_missing"
    if headers.get("Termination", "").strip().lower() != "normal":
        return "non_normal_termination"
    if len(moves) < min_plies:
        return "too_few_plies"
    return None


def _is_rated(headers: chess.pgn.Headers) -> bool:
    rated = headers.get("Rated", "").strip().lower()
    if rated in {"true", "yes", "1"}:
        return True
    if rated in {"false", "no", "0"}:
        return False
    return headers.get("Event", "").strip().lower().startswith("rated ")


def _is_blitz_or_slower(headers: chess.pgn.Headers) -> bool:
    event = headers.get("Event", "").strip().lower()
    if any(label in event for label in ("ultrabullet", "ultra bullet", "bullet")):
        return False
    if any(label in event for label in ("blitz", "rapid", "classical", "correspondence")):
        return True

    time_control = headers.get("TimeControl", "").strip()
    parts = time_control.split("+", maxsplit=1)
    if len(parts) != 2:
        return False
    try:
        initial_seconds = int(parts[0])
        increment_seconds = int(parts[1])
    except ValueError:
        return False
    estimated_seconds = initial_seconds + ESTIMATED_GAME_INCREMENT_MOVES * increment_seconds
    return estimated_seconds >= BLITZ_OR_SLOWER_SECONDS


def _has_minimum_elos(headers: chess.pgn.Headers, min_elo: int) -> bool:
    white_elo = _int_header(headers, "WhiteElo")
    black_elo = _int_header(headers, "BlackElo")
    return white_elo is not None and black_elo is not None and min(white_elo, black_elo) >= min_elo


def _int_header(headers: chess.pgn.Headers, key: str) -> int | None:
    value = headers.get(key, "").strip()
    if not value.isdigit():
        return None
    return int(value)


def _evenly_spaced_indices(*, length: int, limit: int) -> list[int]:
    if length <= 0 or limit <= 0:
        return []
    if limit >= length:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


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


def _game_id(*, game: chess.pgn.Game, moves: list[chess.Move]) -> str:
    headers = game.headers
    identity = {
        "site": headers.get("Site", "?"),
        "date": headers.get("Date", "?"),
        "round": headers.get("Round", "?"),
        "white": headers.get("White", "?"),
        "black": headers.get("Black", "?"),
        "moves": [move.uci() for move in moves],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _source_metadata(game: chess.pgn.Game) -> dict[str, str]:
    headers = game.headers
    return {
        "event": headers.get("Event", "?"),
        "site": headers.get("Site", "?"),
        "date": headers.get("Date", "?"),
        "round": headers.get("Round", "?"),
        "white": headers.get("White", "?"),
        "black": headers.get("Black", "?"),
        "white_elo": headers.get("WhiteElo", "?"),
        "black_elo": headers.get("BlackElo", "?"),
        "time_control": headers.get("TimeControl", "?"),
        "termination": headers.get("Termination", "?"),
    }


def _validate_args(
    *,
    max_examples: int,
    val_ratio: float,
    constrained_ratio: float,
    min_elo: int,
    min_plies: int,
    max_positions_per_game: int,
) -> None:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if not 0 <= constrained_ratio <= 1:
        raise ValueError("constrained_ratio must be between 0 and 1")
    if min_elo < 0:
        raise ValueError("min_elo must be non-negative")
    if min_plies <= 0:
        raise ValueError("min_plies must be positive")
    if max_positions_per_game <= 0:
        raise ValueError("max_positions_per_game must be positive")


@contextmanager
def _open_pgn(pgn: str) -> Iterator[TextIO]:
    if pgn == "-":
        yield sys.stdin
        return
    with Path(pgn).open(encoding="utf-8", errors="replace") as pgn_file:
        yield pgn_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build strict-v7 train/val fine-tuning JSONL files from streamed Lichess PGN."
        )
    )
    parser.add_argument(
        "--pgn",
        required=True,
        help="Input PGN path, or '-' to read PGN from stdin, e.g. zstdcat dump.pgn.zst | ...",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        required=True,
        help="Train JSONL output path, usually under data/finetune/.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        required=True,
        help="Validation JSONL output path, usually under data/finetune/.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional sidecar JSON metadata path.",
    )
    parser.add_argument("--max-examples", type=int, default=20_000)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--constrained-ratio", type=float, default=0.3)
    parser.add_argument("--no-ascii-board", action="store_true")
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--min-plies", type=int, default=20)
    parser.add_argument("--max-positions-per-game", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()

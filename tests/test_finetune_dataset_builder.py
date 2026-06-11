import json
import random
from pathlib import Path

from finetune.build_dataset import build_dataset

LEGAL_20_PLY_GAME = (
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 "
    "6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 *"
)


def test_dataset_builder_filters_and_samples_labeled_positions(tmp_path: Path) -> None:
    pgn = tmp_path / "lichess_sample.pgn"
    train_output = tmp_path / "train.jsonl"
    val_output = tmp_path / "val.jsonl"
    pgn.write_text(
        "\n\n".join(
            [
                _game(
                    event="Rated Bullet game",
                    white_elo=2400,
                    black_elo=2400,
                    time_control="60+0",
                ),
                _game(
                    event="Casual Rapid game",
                    white_elo=2400,
                    black_elo=2400,
                    time_control="600+0",
                ),
                _game(
                    event="Rated Blitz game",
                    white_elo=1999,
                    black_elo=2400,
                    time_control="300+0",
                ),
                _game(
                    event="Rated Blitz game",
                    white_elo=2200,
                    black_elo=2300,
                    time_control="300+0",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = build_dataset(
        pgn_path=pgn,
        train_output_path=train_output,
        val_output_path=val_output,
        max_examples=10,
        val_ratio=0.0,
        constrained_ratio=1.0,
        include_ascii_board=True,
        min_elo=2000,
        min_plies=20,
        max_positions_per_game=4,
        rng=random.Random(0),
    )

    rows = [json.loads(line) for line in train_output.read_text(encoding="utf-8").splitlines()]
    assert val_output.read_text(encoding="utf-8") == ""
    assert stats.games_seen == 4
    assert stats.games_kept == 1
    assert stats.examples_train == 4
    assert stats.skipped_games == {
        "time_control_too_fast_or_unknown": 1,
        "unrated": 1,
        "elo_below_threshold_or_missing": 1,
    }

    assert {row["game_id"] for row in rows} == {rows[0]["game_id"]}
    assert [row["ply"] for row in rows] == [1, 7, 14, 20]
    assert rows[0]["completion"] == '{"move":"e2e4"}'
    assert rows[0]["prompt_version"] == "strict-v7"
    assert rows[0]["legality_mode"] == "constrained"
    assert "Legal moves (UCI):" in rows[0]["prompt"]
    assert rows[0]["source"]["white_elo"] == "2200"


def test_dataset_builder_splits_by_game_not_position(tmp_path: Path) -> None:
    pgn = tmp_path / "two_games.pgn"
    train_output = tmp_path / "train.jsonl"
    val_output = tmp_path / "val.jsonl"
    pgn.write_text(
        "\n\n".join(
            [
                _game(
                    event="Rated Blitz game",
                    white="TeacherA",
                    black="TeacherB",
                    white_elo=2200,
                    black_elo=2300,
                    time_control="300+0",
                ),
                _game(
                    event="Rated Rapid game",
                    white="TeacherC",
                    black="TeacherD",
                    white_elo=2250,
                    black_elo=2350,
                    time_control="600+0",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    stats = build_dataset(
        pgn_path=pgn,
        train_output_path=train_output,
        val_output_path=val_output,
        max_examples=10,
        val_ratio=1.0,
        constrained_ratio=0.0,
        include_ascii_board=False,
        min_elo=2000,
        min_plies=20,
        max_positions_per_game=3,
        rng=random.Random(0),
    )

    train_rows = train_output.read_text(encoding="utf-8").splitlines()
    val_rows = [json.loads(line) for line in val_output.read_text(encoding="utf-8").splitlines()]
    assert train_rows == []
    assert stats.games_kept == 2
    assert stats.examples_val == 6
    assert {row["split"] for row in val_rows} == {"val"}
    assert all(row["legality_mode"] == "open" for row in val_rows)
    assert all("ASCII board:" not in row["prompt"] for row in val_rows)

    first_game_ids = {row["game_id"] for row in val_rows[:3]}
    second_game_ids = {row["game_id"] for row in val_rows[3:]}
    assert len(first_game_ids) == 1
    assert len(second_game_ids) == 1
    assert first_game_ids != second_game_ids


def _game(
    *,
    event: str,
    white_elo: int,
    black_elo: int,
    time_control: str,
    white: str = "TeacherWhite",
    black: str = "TeacherBlack",
    termination: str = "Normal",
) -> str:
    return "\n".join(
        [
            f'[Event "{event}"]',
            '[Site "https://lichess.org/test"]',
            '[Date "2026.06.11"]',
            '[Round "-"]',
            f'[White "{white}"]',
            f'[Black "{black}"]',
            f'[WhiteElo "{white_elo}"]',
            f'[BlackElo "{black_elo}"]',
            f'[TimeControl "{time_control}"]',
            f'[Termination "{termination}"]',
            '[Result "*"]',
            "",
            LEGAL_20_PLY_GAME,
        ]
    )

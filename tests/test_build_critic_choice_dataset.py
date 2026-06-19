from __future__ import annotations

import json
from pathlib import Path

from finetune.build_critic_choice_dataset import ChoiceDatasetConfig, build_choice_dataset

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SECOND_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_build_critic_choice_dataset_writes_best_move_examples(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "choice.jsonl"
    _write_rows(
        input_path,
        [
            _row(START_FEN, "e2e4", "e4", "playable", 80, 80),
            _row(START_FEN, "g1f3", "Nf3", "good", 100, 0),
            _row(START_FEN, "f2f3", "f3", "blunder", 0, 500),
            _row(SECOND_FEN, "e7e5", "e5", "good", 100, 0),
            _row(SECOND_FEN, "c7c5", "c5", "playable", 80, 80),
        ],
    )

    stats = build_choice_dataset(
        ChoiceDatasetConfig(
            input=str(input_path),
            output=str(output_path),
            metadata_output=None,
            min_candidates=2,
            max_candidates=3,
            require_mixed_risk=True,
        )
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.positions_seen == 2
    assert stats.positions_written == 1
    assert stats.dropped_no_mixed_risk == 1
    assert rows[0]["fen"] == START_FEN
    assert rows[0]["move"] == "g1f3"
    assert rows[0]["completion"] == '{"move":"g1f3"}'
    assert rows[0]["target_centipawn_loss"] == 0
    assert rows[0]["target_prompt_index"] >= 1
    assert rows[0]["candidate_count"] == 3
    assert "Candidate moves:" in rows[0]["prompt"]
    assert "g1f3 (Nf3)" in rows[0]["prompt"]
    assert any(candidate["uci"] == "g1f3" for candidate in rows[0]["candidates"])
    assert stats.by_target_prompt_index[rows[0]["target_prompt_index"]] == 1


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(
    fen: str,
    move: str,
    san: str,
    risk: str,
    score: int,
    cpl: int,
) -> dict[str, object]:
    return {
        "fen_before": fen,
        "side_to_move": "white",
        "candidate_uci": move,
        "candidate_san": san,
        "source": "arena_candidate",
        "risk": risk,
        "score": score,
        "centipawn_loss": cpl,
        "candidate_rank_in_generator": None,
    }

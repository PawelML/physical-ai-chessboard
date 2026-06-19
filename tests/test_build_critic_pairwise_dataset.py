from __future__ import annotations

import json
from pathlib import Path

from finetune.build_critic_pairwise_dataset import (
    PairwiseDatasetConfig,
    build_pairwise_dataset,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SECOND_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_build_pairwise_dataset_writes_safe_vs_unsafe_pairs(tmp_path: Path) -> None:
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "pairwise.jsonl"
    metadata_path = tmp_path / "pairwise.meta.json"
    _write_rows(
        input_path,
        [
            _row(START_FEN, "e2e4", "e4", "playable", 80, 80),
            _row(START_FEN, "g1f3", "Nf3", "good", 100, 0),
            _row(START_FEN, "f2f3", "f3", "blunder", 0, 500),
            _row(START_FEN, "b1c3", "Nc3", "mistake", 45, 220),
            _row(SECOND_FEN, "e7e5", "e5", "good", 100, 0),
            _row(SECOND_FEN, "c7c5", "c5", "playable", 80, 80),
        ],
    )

    stats = build_pairwise_dataset(
        PairwiseDatasetConfig(
            input=str(input_path),
            output=str(output_path),
            metadata_output=str(metadata_path),
            pairs_per_position=2,
            min_cpl_gap=100,
        )
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.positions_seen == 2
    assert stats.positions_with_pairs == 1
    assert stats.examples_written == 2
    assert stats.dropped_no_unsafe_candidate == 1
    assert stats.by_target_risk == {"good": 1, "playable": 1}
    assert stats.by_target_prompt_index[1] + stats.by_target_prompt_index[2] == 2
    assert rows[0]["fen"] == START_FEN
    assert rows[0]["completion"] == f'{{"move":"{rows[0]["move"]}"}}'
    assert rows[0]["candidate_count"] == 2
    assert rows[0]["target_prompt_index"] in {1, 2}
    assert rows[0]["target_risk"] in {"good", "playable"}
    assert rows[0]["pair_cpl_gap"] >= 100
    assert "Choose the safer move from the two candidates." in rows[0]["prompt"]
    assert len(rows[0]["candidates"]) == 2


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

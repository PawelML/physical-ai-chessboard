from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from arena_core.llm.base import GenerationOptions, LLMResponse, LLMService
from finetune.sample_policy_candidates import (
    PolicyCandidateSampleConfig,
    sample_policy_candidates,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
WHITE_AFTER_E4_E5_FEN = (
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
)


async def test_sample_policy_candidates_writes_legal_unique_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "choice.jsonl"
    output_path = tmp_path / "policy_candidates.jsonl"
    input_path.write_text(
        json.dumps({"fen": START_FEN, "split": "train"}) + "\n",
        encoding="utf-8",
    )
    service = FakeLLMService(
        json.dumps(
            {
                "candidates": [
                    {"move": "e2e4", "idea": "claim center"},
                    {"move": "e2e5", "idea": "illegal"},
                    {"move": "g1f3", "idea": "develop"},
                    {"move": "e2e4", "idea": "duplicate"},
                ]
            }
        )
    )

    stats = await sample_policy_candidates(
        PolicyCandidateSampleConfig(
            input=str(input_path),
            output=str(output_path),
            metadata_output=None,
            model="policy-model",
            split="train",
            samples_per_position=1,
            n_candidates=5,
        ),
        service=service,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.positions_seen == 1
    assert stats.positions_sampled == 1
    assert stats.requests == 1
    assert stats.legal_candidates_seen == 2
    assert stats.rows_written == 2
    assert [row["candidate_uci"] for row in rows] == ["e2e4", "g1f3"]
    assert rows[0]["source"] == "policy_sample"
    assert rows[0]["candidate_rank_in_generator"] == 0
    assert rows[0]["generator_idea"] == "claim center"
    assert rows[0]["policy_model"] == "policy-model"
    assert rows[1]["candidate_rank_in_generator"] == 2


async def test_sample_policy_candidates_reads_arena_db_positions(tmp_path: Path) -> None:
    db_path = tmp_path / "arena.db"
    output_path = tmp_path / "policy_candidates.jsonl"
    _write_sample_arena_db(db_path)
    service = FakeLLMService(
        json.dumps(
            {
                "candidates": [
                    {"move": "e2e4", "idea": "claim center"},
                    {"move": "g1f3", "idea": "develop"},
                ]
            }
        )
    )

    stats = await sample_policy_candidates(
        PolicyCandidateSampleConfig(
            input=None,
            arena_db=str(db_path),
            run_ids=[7],
            output=str(output_path),
            metadata_output=None,
            model="policy-model",
            split=None,
            samples_per_position=1,
            n_candidates=5,
        ),
        service=service,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.arena_db_rows == 3
    assert stats.positions_seen == 2
    assert stats.positions_from_arena_db == 2
    assert stats.requests == 2
    assert stats.rows_written == 3
    assert [row["fen_before"] for row in rows] == [
        START_FEN,
        START_FEN,
        WHITE_AFTER_E4_E5_FEN,
    ]
    assert [row["candidate_uci"] for row in rows] == ["e2e4", "g1f3", "g1f3"]


async def test_sample_policy_candidates_offsets_unique_positions(tmp_path: Path) -> None:
    db_path = tmp_path / "arena.db"
    output_path = tmp_path / "policy_candidates.jsonl"
    _write_sample_arena_db(db_path)
    service = FakeLLMService(json.dumps({"candidates": [{"move": "g1f3"}]}))

    stats = await sample_policy_candidates(
        PolicyCandidateSampleConfig(
            input=None,
            arena_db=str(db_path),
            run_ids=[7],
            output=str(output_path),
            metadata_output=None,
            model="policy-model",
            split=None,
            position_offset=1,
            max_positions=1,
            samples_per_position=1,
            n_candidates=5,
        ),
        service=service,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert stats.positions_skipped_by_offset == 1
    assert stats.positions_seen == 1
    assert stats.positions_from_arena_db == 1
    assert stats.requests == 1
    assert [row["fen_before"] for row in rows] == [WHITE_AFTER_E4_E5_FEN]
    assert [row["candidate_uci"] for row in rows] == ["g1f3"]


def _write_sample_arena_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            create table games (
                id integer primary key,
                run_id integer not null
            );
            create table moves (
                id integer primary key,
                game_id integer not null,
                ply integer not null,
                fen_before text not null
            );
            """
        )
        con.execute("insert into games (id, run_id) values (1, 7)")
        con.execute("insert into games (id, run_id) values (2, 8)")
        con.executemany(
            "insert into moves (id, game_id, ply, fen_before) values (?, ?, ?, ?)",
            [
                (10, 1, 1, START_FEN),
                (11, 1, 2, START_FEN),
                (12, 1, 3, WHITE_AFTER_E4_E5_FEN),
                (13, 2, 1, START_FEN),
            ],
        )
        con.commit()
    finally:
        con.close()


class FakeLLMService(LLMService):
    def __init__(self, response: str) -> None:
        self.response = response

    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse:
        del model, prompt, options
        return LLMResponse(content=self.response)

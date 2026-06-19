from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import chess

from arena_core.evaluators.stockfish import EngineEvaluation
from finetune.build_critic_ranker_dataset import (
    CachedMoveEvaluator,
    CriticRankerConfig,
    build_critic_ranker_dataset,
    risk_from_evaluation,
    score_from_evaluation,
    split_for_fen,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_build_critic_ranker_dataset_writes_candidate_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "arena.db"
    output_path = tmp_path / "critic.jsonl"
    _write_minimal_arena_db(db_path)
    evaluator = FakeEvaluator(
        {
            "e2e4": _evaluation(move="e2e4", cpl=120, classification="inaccuracy"),
            "g1f3": _evaluation(move="g1f3", cpl=0, classification="best"),
        }
    )

    stats = build_critic_ranker_dataset(
        CriticRankerConfig(
            db=str(db_path),
            output=str(output_path),
            metadata_output=None,
            stockfish_path="/unused",
            run_ids=[7],
            max_candidates_per_position=6,
            random_legal_per_position=0,
            include_stockfish_best=True,
        ),
        evaluator=evaluator,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_move = {row["candidate_uci"]: row for row in rows}

    assert stats.positions_written == 1
    assert stats.rows_written == 2
    assert stats.dropped_duplicate_candidate == 2
    assert evaluator.calls == 2
    assert set(by_move) == {"e2e4", "g1f3"}
    assert by_move["e2e4"]["source"] == "arena_move"
    assert by_move["e2e4"]["risk"] == "playable"
    assert by_move["e2e4"]["score"] == 70
    assert by_move["g1f3"]["source"] == "arena_candidate"
    assert by_move["g1f3"]["risk"] == "good"
    assert by_move["g1f3"]["generator_idea"] == "Develop the knight."
    assert json.loads(by_move["g1f3"]["completion"]) == {
        "risk": "good",
        "blunder": False,
        "score": 100,
        "reason_type": "none",
    }
    assert "FEN before:" in by_move["g1f3"]["prompt"]
    assert by_move["g1f3"]["split"] == split_for_fen(START_FEN)


def test_risk_and_score_bucket_engine_evaluations() -> None:
    assert risk_from_evaluation(_evaluation(move="e2e4", cpl=0, classification="best")) == "good"
    assert (
        risk_from_evaluation(_evaluation(move="e2e4", cpl=75, classification="inaccuracy"))
        == "playable"
    )
    assert (
        risk_from_evaluation(_evaluation(move="e2e4", cpl=250, classification="mistake"))
        == "mistake"
    )
    assert (
        risk_from_evaluation(_evaluation(move="e2e4", cpl=350, classification="blunder"))
        == "blunder"
    )
    assert score_from_evaluation(_evaluation(move="e2e4", cpl=350, classification="blunder")) == 12
    assert (
        score_from_evaluation(_evaluation(move="e2e4", cpl=None, classification="mate_missed"))
        == 0
    )


def test_cached_move_evaluator_reuses_persisted_evaluations(tmp_path: Path) -> None:
    cache_path = tmp_path / "eval_cache.jsonl"
    board = chess.Board(START_FEN)
    move = chess.Move.from_uci("e2e4")
    first_inner = FakeEvaluator(
        {"e2e4": _evaluation(move="e2e4", cpl=120, classification="inaccuracy")}
    )
    first = CachedMoveEvaluator(first_inner, cache_path=cache_path)

    first_eval = first.evaluate_move(board, move)
    second_eval = first.evaluate_move(board, move)
    first.close()

    assert first_eval.centipawn_loss == 120
    assert second_eval.centipawn_loss == 120
    assert first_inner.calls == 1
    assert first.cache_misses == 1
    assert first.cache_hits == 1

    second_inner = FakeEvaluator(
        {"e2e4": _evaluation(move="e2e4", cpl=999, classification="blunder")}
    )
    second = CachedMoveEvaluator(second_inner, cache_path=cache_path)
    cached_eval = second.evaluate_move(board, move)
    second.close()

    assert cached_eval.centipawn_loss == 120
    assert second_inner.calls == 0
    assert second.cache_hits == 1


class FakeEvaluator:
    def __init__(self, evaluations: dict[str, EngineEvaluation]) -> None:
        self.evaluations = evaluations
        self.calls = 0

    def evaluate_move(self, board_before: chess.Board, move: chess.Move) -> EngineEvaluation:
        del board_before
        self.calls += 1
        return self.evaluations[move.uci()]

    def close(self) -> None:
        pass


def _write_minimal_arena_db(path: Path) -> None:
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
                fen_before text not null,
                accepted_uci text not null
            );
            create table attempts (
                id integer primary key,
                move_id integer,
                game_id integer not null,
                ply integer not null,
                attempt_number integer not null,
                reranker_metadata text
            );
            create table engine_evaluations (
                id integer primary key,
                move_id integer not null,
                best_move_uci text,
                classification text
            );
            """
        )
        metadata = {
            "regime": "llm_deliberative",
            "candidate_legal_moves": ["g1f3", "e2e4"],
            "candidate_raw_response": json.dumps(
                {"candidates": [{"move": "g1f3", "idea": "Develop the knight."}]}
            ),
        }
        con.execute("insert into games (id, run_id) values (1, 7)")
        con.execute(
            "insert into moves (id, game_id, ply, fen_before, accepted_uci) values (?, ?, ?, ?, ?)",
            (10, 1, 1, START_FEN, "e2e4"),
        )
        con.execute(
            "insert into attempts "
            "(id, move_id, game_id, ply, attempt_number, reranker_metadata) "
            "values (?, ?, ?, ?, ?, ?)",
            (20, 10, 1, 1, 1, json.dumps(metadata)),
        )
        con.execute(
            "insert into engine_evaluations "
            "(id, move_id, best_move_uci, classification) values (?, ?, ?, ?)",
            (30, 10, "g1f3", "inaccuracy"),
        )
        con.commit()
    finally:
        con.close()


def _evaluation(
    *,
    move: str,
    cpl: int | None,
    classification: str,
) -> EngineEvaluation:
    return EngineEvaluation(
        engine_name="stockfish",
        engine_version="fake",
        nodes=1,
        depth_reached=1,
        eval_before_cp=0,
        eval_after_cp=0,
        mate_before=None,
        mate_after=None,
        best_move_uci=move,
        centipawn_loss=cpl,
        classification=classification,
    )

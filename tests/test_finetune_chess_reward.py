from __future__ import annotations

import chess

from arena_core.evaluators.stockfish import EngineEvaluation
from finetune.chess_reward import (
    ILLEGAL_REWARD,
    MALFORMED_REWARD,
    TACTICAL_ILLEGAL_REWARD,
    RewardConfig,
    StockfishRewardScorer,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_stockfish_reward_orders_malformed_illegal_and_legal_moves() -> None:
    evaluator = FakeEvaluator({"e2e4": _evaluation(move="e2e4", cpl=30, best="e2e4")})
    scorer = StockfishRewardScorer(
        RewardConfig(stockfish_path="/unused", workers=1),
        evaluator_factory=lambda: evaluator,
    )

    samples = scorer.score_batch(
        fens=[START_FEN, START_FEN, START_FEN],
        completions=[
            "not json",
            '{"move":"e2e5"}',
            '{"move":"e2e4"}',
        ],
    )

    assert [sample.reward for sample in samples] == [MALFORMED_REWARD, ILLEGAL_REWARD, 0.9]
    assert [sample.parse_ok for sample in samples] == [False, True, True]
    assert [sample.legal_ok for sample in samples] == [False, False, True]
    assert samples[2].centipawn_loss == 30
    assert scorer.latest_stats.to_json()["mean_legal_cpl"] == 30


def test_stockfish_reward_caches_engine_results_by_fen_and_move() -> None:
    evaluator = FakeEvaluator({"e2e4": _evaluation(move="e2e4", cpl=0, best="e2e4")})
    scorer = StockfishRewardScorer(
        RewardConfig(stockfish_path="/unused", workers=1),
        evaluator_factory=lambda: evaluator,
    )

    first = scorer.score_one(fen=START_FEN, completion='{"move":"e2e4"}')
    second = scorer.score_one(fen=START_FEN, completion='{"move":"e2e4"}')

    assert first.reward == 1.0
    assert second.reward == 1.0
    assert evaluator.calls == 1
    assert scorer.latest_stats.cache_hits == 1


def test_stockfish_reward_handles_mate_scores_without_cpl() -> None:
    evaluator = FakeEvaluator(
        {
            "e2e4": _evaluation(move="e2e4", cpl=None, best="g1f3"),
            "g1f3": _evaluation(move="g1f3", cpl=None, best="g1f3"),
        }
    )
    scorer = StockfishRewardScorer(
        RewardConfig(stockfish_path="/unused", workers=1),
        evaluator_factory=lambda: evaluator,
    )

    samples = scorer.score_batch(
        fens=[START_FEN, START_FEN],
        completions=['{"move":"e2e4"}', '{"move":"g1f3"}'],
    )

    assert [sample.reward for sample in samples] == [0.0, 1.0]


def test_tactical_reward_penalizes_missed_mates_without_cpl() -> None:
    evaluator = FakeEvaluator(
        {
            "e2e4": _evaluation(move="e2e4", cpl=None, best="g1f3"),
            "g1f3": _evaluation(move="g1f3", cpl=None, best="g1f3"),
        }
    )
    scorer = StockfishRewardScorer(
        RewardConfig(stockfish_path="/unused", workers=1, reward_mode="tactical"),
        evaluator_factory=lambda: evaluator,
    )

    samples = scorer.score_batch(
        fens=[START_FEN, START_FEN],
        completions=['{"move":"e2e4"}', '{"move":"g1f3"}'],
    )

    assert [sample.reward for sample in samples] == [-1.0, 1.0]


def test_tactical_reward_penalizes_high_cpl_legal_moves() -> None:
    evaluator = FakeEvaluator(
        {
            "g1f3": _evaluation(move="g1f3", cpl=20, best="g1f3"),
            "e2e4": _evaluation(move="e2e4", cpl=300, best="g1f3"),
            "b1c3": _evaluation(move="b1c3", cpl=450, best="g1f3"),
        }
    )
    scorer = StockfishRewardScorer(
        RewardConfig(stockfish_path="/unused", workers=1, reward_mode="tactical"),
        evaluator_factory=lambda: evaluator,
    )

    samples = scorer.score_batch(
        fens=[START_FEN, START_FEN, START_FEN, START_FEN],
        completions=[
            '{"move":"g1f3"}',
            '{"move":"e2e4"}',
            '{"move":"b1c3"}',
            '{"move":"e2e5"}',
        ],
    )

    assert [sample.reward for sample in samples] == [
        1.0,
        -0.8,
        -1.0,
        TACTICAL_ILLEGAL_REWARD,
    ]
    assert samples[1].classification == "mate_missed"
    assert samples[2].centipawn_loss == 450


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


def _evaluation(*, move: str, cpl: int | None, best: str | None) -> EngineEvaluation:
    return EngineEvaluation(
        engine_name="stockfish",
        engine_version="fake",
        nodes=1,
        depth_reached=1,
        eval_before_cp=0,
        eval_after_cp=0,
        mate_before=None if cpl is not None else 1,
        mate_after=None,
        best_move_uci=best,
        centipawn_loss=cpl,
        classification="best" if move == best else "mate_missed",
    )

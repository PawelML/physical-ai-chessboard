from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import chess

from arena_core.evaluators.stockfish import EngineEvaluation, StockfishEvaluator
from arena_core.parser import parse_uci_json

CPL_CAP = 300
MALFORMED_REWARD = -1.0
ILLEGAL_REWARD = -0.5
TACTICAL_ILLEGAL_REWARD = -1.0

_WORKER_EVALUATOR: StockfishEvaluator | None = None

EvaluatorFactory = Callable[[], StockfishEvaluator]


@dataclass(frozen=True)
class RewardConfig:
    stockfish_path: str
    reward_nodes: int = 50_000
    workers: int = 1
    hash_mb: int = 64
    cpl_cap: int = CPL_CAP
    reward_mode: str = "linear"
    tactical_good_cpl: int = 80
    tactical_inaccuracy_cpl: int = 150
    tactical_blunder_cpl: int = 300


@dataclass(frozen=True)
class RewardSample:
    reward: float
    parse_ok: bool
    legal_ok: bool
    parsed_move: str | None
    centipawn_loss: int | None
    best_move_uci: str | None
    classification: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RewardBatchStats:
    examples: int = 0
    malformed: int = 0
    illegal: int = 0
    legal: int = 0
    reward_sum: float = 0.0
    cpl_sum: int = 0
    cpl_count: int = 0
    engine_evaluations: int = 0
    cache_hits: int = 0

    def to_json(self) -> dict[str, object]:
        return {
            "examples": self.examples,
            "malformed": self.malformed,
            "illegal": self.illegal,
            "legal": self.legal,
            "malformed_fraction": _rate(self.malformed, self.examples),
            "illegal_fraction": _rate(self.illegal, self.examples),
            "legal_fraction": _rate(self.legal, self.examples),
            "mean_reward": self.reward_sum / self.examples if self.examples else 0.0,
            "mean_legal_cpl": self.cpl_sum / self.cpl_count if self.cpl_count else None,
            "engine_evaluations": self.engine_evaluations,
            "cache_hits": self.cache_hits,
        }


class StockfishRewardScorer:
    __name__ = "stockfish_cpl_reward"

    def __init__(
        self,
        config: RewardConfig,
        *,
        evaluator_factory: EvaluatorFactory | None = None,
        metrics_output: Path | None = None,
    ) -> None:
        if config.reward_mode not in {"linear", "tactical"}:
            raise ValueError("reward_mode must be 'linear' or 'tactical'")
        self.config = config
        self.metrics_output = metrics_output
        self.latest_stats = RewardBatchStats()
        self._evaluator_factory = evaluator_factory
        self._evaluator: StockfishEvaluator | None = None
        self._pool: Pool | None = None
        self._cache: dict[tuple[str, str], RewardSample] = {}
        self._best_move_cache: dict[str, str | None] = {}

    def __call__(
        self,
        *,
        prompts: list[Any],
        completions: list[Any],
        fen: list[str],
        trainer_state: Any | None = None,
        **kwargs: object,
    ) -> list[float]:
        del prompts, kwargs
        samples = self.score_batch(fens=fen, completions=completions)
        self._record_metrics(samples=samples, trainer_state=trainer_state)
        return [sample.reward for sample in samples]

    def score_batch(self, *, fens: list[str], completions: list[Any]) -> list[RewardSample]:
        if len(fens) != len(completions):
            raise ValueError("fen and completions must have the same length")

        parsed_items: list[tuple[str, str, chess.Move] | None] = []
        samples: list[RewardSample | None] = []
        tasks: list[tuple[str, str]] = []
        task_indexes: list[int] = []
        stats = RewardBatchStats(examples=len(completions))

        for index, (fen, completion) in enumerate(zip(fens, completions, strict=True)):
            prepared = self._prepare_completion(fen=fen, completion=completion)
            parsed_items.append(prepared)
            if isinstance(prepared, RewardSample):
                samples.append(prepared)
                if not prepared.parse_ok:
                    stats.malformed += 1
                elif not prepared.legal_ok:
                    stats.illegal += 1
                continue

            assert prepared is not None
            _, uci, _ = prepared
            cached = self._cache.get((fen, uci))
            if cached is not None:
                samples.append(cached)
                stats.cache_hits += 1
                continue
            best_move = self._best_move_cache.get(fen)
            if best_move is not None and best_move == uci:
                sample = RewardSample(
                    reward=1.0,
                    parse_ok=True,
                    legal_ok=True,
                    parsed_move=uci,
                    centipawn_loss=0,
                    best_move_uci=best_move,
                    classification="best",
                )
                self._cache[(fen, uci)] = sample
                samples.append(sample)
                stats.cache_hits += 1
                continue
            samples.append(None)
            tasks.append((fen, uci))
            task_indexes.append(index)

        scored_tasks = self._score_engine_tasks(tasks)
        stats.engine_evaluations = len(tasks)
        for sample_index, sample in zip(task_indexes, scored_tasks, strict=True):
            prepared = parsed_items[sample_index]
            assert not isinstance(prepared, RewardSample) and prepared is not None
            fen, uci, _ = prepared
            self._cache[(fen, uci)] = sample
            self._best_move_cache[fen] = sample.best_move_uci
            samples[sample_index] = sample

        if any(sample is None for sample in samples):
            raise RuntimeError("Reward batch scoring left unfilled samples")
        finalized = [sample for sample in samples if sample is not None]
        for sample in finalized:
            stats.reward_sum += sample.reward
            if sample.legal_ok:
                stats.legal += 1
                if sample.centipawn_loss is not None:
                    stats.cpl_sum += sample.centipawn_loss
                    stats.cpl_count += 1
        self.latest_stats = stats
        return finalized

    def score_one(self, *, fen: str, completion: Any) -> RewardSample:
        return self.score_batch(fens=[fen], completions=[completion])[0]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
        if self._evaluator is not None:
            self._evaluator.close()
            self._evaluator = None

    def _prepare_completion(
        self,
        *,
        fen: str,
        completion: Any,
    ) -> tuple[str, str, chess.Move] | RewardSample:
        raw_completion = _completion_text(completion)
        parsed = parse_uci_json(raw_completion)
        if not parsed.parse_ok or parsed.move is None:
            return RewardSample(
                reward=MALFORMED_REWARD,
                parse_ok=False,
                legal_ok=False,
                parsed_move=parsed.move,
                centipawn_loss=None,
                best_move_uci=None,
                classification="malformed",
            )

        try:
            move = chess.Move.from_uci(parsed.move)
            board = chess.Board(fen)
        except ValueError:
            return RewardSample(
                reward=self._illegal_reward(),
                parse_ok=True,
                legal_ok=False,
                parsed_move=parsed.move,
                centipawn_loss=None,
                best_move_uci=None,
                classification="illegal",
            )
        if move not in board.legal_moves:
            return RewardSample(
                reward=self._illegal_reward(),
                parse_ok=True,
                legal_ok=False,
                parsed_move=parsed.move,
                centipawn_loss=None,
                best_move_uci=None,
                classification="illegal",
            )
        return fen, move.uci(), move

    def _score_engine_tasks(self, tasks: list[tuple[str, str]]) -> list[RewardSample]:
        if not tasks:
            return []
        if self.config.workers <= 1 or self._evaluator_factory is not None:
            evaluator = self._ensure_evaluator()
            return [_score_with_evaluator(evaluator, task, self.config) for task in tasks]
        pool = self._ensure_pool()
        payload = [(fen, uci, self.config) for fen, uci in tasks]
        return list(pool.imap(_score_worker_task, payload, chunksize=8))

    def _illegal_reward(self) -> float:
        if self.config.reward_mode == "tactical":
            return TACTICAL_ILLEGAL_REWARD
        return ILLEGAL_REWARD

    def _ensure_evaluator(self) -> StockfishEvaluator:
        if self._evaluator is None:
            if self._evaluator_factory is not None:
                self._evaluator = self._evaluator_factory()
            else:
                self._evaluator = StockfishEvaluator(
                    binary_path=self.config.stockfish_path,
                    nodes=self.config.reward_nodes,
                    threads=1,
                    hash_mb=self.config.hash_mb,
                )
        return self._evaluator

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = Pool(
                processes=self.config.workers,
                initializer=_init_worker,
                initargs=(
                    self.config.stockfish_path,
                    self.config.reward_nodes,
                    self.config.hash_mb,
                ),
            )
        return self._pool

    def _record_metrics(
        self,
        *,
        samples: list[RewardSample],
        trainer_state: Any | None,
    ) -> None:
        if self.metrics_output is None:
            return
        self.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        payload = self.latest_stats.to_json()
        payload["timestamp"] = time.time()
        if trainer_state is not None:
            payload["global_step"] = getattr(trainer_state, "global_step", None)
        payload["samples"] = [sample.to_json() for sample in samples[:5]]
        with self.metrics_output.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def __enter__(self) -> StockfishRewardScorer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def default_stockfish_path() -> str | None:
    return os.environ.get("ARENA_STOCKFISH_PATH")


def require_stockfish_path(path: str | None) -> str:
    resolved = path or default_stockfish_path()
    if not resolved:
        raise SystemExit("No Stockfish binary: pass --stockfish-path or set ARENA_STOCKFISH_PATH.")
    return resolved


def _init_worker(stockfish_path: str, nodes: int, hash_mb: int) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = StockfishEvaluator(
        binary_path=stockfish_path,
        nodes=nodes,
        threads=1,
        hash_mb=hash_mb,
    )


def _score_worker_task(payload: tuple[str, str, RewardConfig]) -> RewardSample:
    assert _WORKER_EVALUATOR is not None
    fen, uci, config = payload
    return _score_with_evaluator(_WORKER_EVALUATOR, (fen, uci), config)


def _score_with_evaluator(
    evaluator: StockfishEvaluator,
    task: tuple[str, str],
    config: RewardConfig,
) -> RewardSample:
    fen, uci = task
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    evaluation = evaluator.evaluate_move(board, move)
    return _sample_from_evaluation(evaluation=evaluation, parsed_move=uci, config=config)


def _sample_from_evaluation(
    *,
    evaluation: EngineEvaluation,
    parsed_move: str,
    config: RewardConfig,
) -> RewardSample:
    if evaluation.centipawn_loss is None:
        reward = _mate_or_unknown_reward(evaluation, parsed_move, config)
    elif config.reward_mode == "tactical":
        reward = _tactical_cpl_reward(evaluation.centipawn_loss, config)
    else:
        reward = 1.0 - min(evaluation.centipawn_loss, config.cpl_cap) / config.cpl_cap
    return RewardSample(
        reward=reward,
        parse_ok=True,
        legal_ok=True,
        parsed_move=parsed_move,
        centipawn_loss=evaluation.centipawn_loss,
        best_move_uci=evaluation.best_move_uci,
        classification=evaluation.classification,
    )


def _mate_or_unknown_reward(
    evaluation: EngineEvaluation,
    parsed_move: str,
    config: RewardConfig,
) -> float:
    if parsed_move == evaluation.best_move_uci:
        return 1.0
    if config.reward_mode == "tactical" and evaluation.classification == "mate_missed":
        return -1.0
    return 0.0


def _tactical_cpl_reward(cpl: int, config: RewardConfig) -> float:
    if cpl <= 20:
        return 1.0
    if cpl <= config.tactical_good_cpl:
        return _lerp_by_cpl(cpl, 20, config.tactical_good_cpl, 1.0, 0.6)
    if cpl <= config.tactical_inaccuracy_cpl:
        return _lerp_by_cpl(
            cpl,
            config.tactical_good_cpl,
            config.tactical_inaccuracy_cpl,
            0.6,
            0.1,
        )
    if cpl <= config.tactical_blunder_cpl:
        return _lerp_by_cpl(
            cpl,
            config.tactical_inaccuracy_cpl,
            config.tactical_blunder_cpl,
            0.1,
            -0.8,
        )
    return -1.0


def _lerp_by_cpl(
    cpl: int,
    lower_cpl: int,
    upper_cpl: int,
    lower_reward: float,
    upper_reward: float,
) -> float:
    if upper_cpl <= lower_cpl:
        return upper_reward
    fraction = (cpl - lower_cpl) / (upper_cpl - lower_cpl)
    return lower_reward + fraction * (upper_reward - lower_reward)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return _completion_from_messages(completion)
    return str(completion)


def _completion_from_messages(messages: Iterable[Any]) -> str:
    content_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            content_parts.append(str(message["content"]))
    return "\n".join(content_parts)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0

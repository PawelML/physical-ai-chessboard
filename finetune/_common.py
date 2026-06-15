import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TextIO

import chess

from arena_core.evaluators.stockfish import StockfishEvaluator

_WORKER_EVALUATOR: StockfishEvaluator | None = None


def is_legal_move(*, fen: str, move_text: str | None) -> bool:
    if move_text is None:
        return False
    try:
        move = chess.Move.from_uci(move_text)
    except ValueError:
        return False
    return move in chess.Board(fen).legal_moves


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def update_eval_stats(stats: Any, prediction: dict[str, Any]) -> None:
    stats.examples += 1
    stats.json_parse_ok += int(prediction["parse_ok"])
    stats.legal_move_ok += int(prediction["legal_ok"])
    stats.top1_match += int(prediction["top1_match"])


def run_prediction_eval(
    *,
    dataset_path: Path,
    limit: int | None,
    predictions_output_path: Path | None,
    stats: Any,
    predict: Callable[[dict[str, Any]], dict[str, Any]],
    update_stats: Callable[[Any, dict[str, Any]], None] = update_eval_stats,
) -> Any:
    predictions_file = _open_optional_predictions(predictions_output_path)
    try:
        with dataset_path.open(encoding="utf-8") as dataset_file:
            for line in dataset_file:
                if limit is not None and stats.examples >= limit:
                    break
                prediction = predict(json.loads(line))
                update_stats(stats, prediction)
                _write_prediction(predictions_file, prediction)
    finally:
        if predictions_file is not None:
            predictions_file.close()
    return stats


async def run_async_prediction_eval(
    *,
    dataset_path: Path,
    limit: int | None,
    predictions_output_path: Path | None,
    stats: Any,
    predict: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    update_stats: Callable[[Any, dict[str, Any]], None] = update_eval_stats,
    progress_every: int | None = None,
    progress: Callable[[Any], None] | None = None,
) -> Any:
    predictions_file = _open_optional_predictions(predictions_output_path)
    try:
        with dataset_path.open(encoding="utf-8") as dataset_file:
            for line in dataset_file:
                if limit is not None and stats.examples >= limit:
                    break
                prediction = await predict(json.loads(line))
                update_stats(stats, prediction)
                _write_prediction(predictions_file, prediction)
                if progress_every is not None and progress is not None:
                    if stats.examples % progress_every == 0:
                        progress(stats)
    finally:
        if predictions_file is not None:
            predictions_file.close()
    return stats


def _open_optional_predictions(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _write_prediction(predictions_file: TextIO | None, prediction: dict[str, Any]) -> None:
    if predictions_file is not None:
        predictions_file.write(json.dumps(prediction, separators=(",", ":")) + "\n")


def write_metadata_sidecar(path: Path, *, config: object, stats: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"config": asdict(config), "stats": stats.to_json()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_metrics_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config_from_args[ConfigT](
    config_type: type[ConfigT],
    args: object,
    *,
    overrides: dict[str, Any] | None = None,
) -> ConfigT:
    if not is_dataclass(config_type):
        raise TypeError("config_type must be a dataclass")
    override_values = overrides or {}
    values: dict[str, Any] = {}
    for field in fields(config_type):
        if field.name in override_values:
            value = override_values[field.name]
        else:
            value = getattr(args, field.name)
        if isinstance(value, Path):
            value = str(value)
        values[field.name] = value
    return config_type(**values)


def init_stockfish_worker(stockfish_path: str, nodes: int, hash_mb: int) -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = StockfishEvaluator(
        binary_path=stockfish_path,
        nodes=nodes,
        threads=1,
        hash_mb=hash_mb,
    )


def stockfish_worker_evaluator() -> StockfishEvaluator:
    assert _WORKER_EVALUATOR is not None
    return _WORKER_EVALUATOR

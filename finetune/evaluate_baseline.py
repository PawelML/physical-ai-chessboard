from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arena_core.llm.base import LLMResponse, LLMService
from arena_core.llm.ollama import OllamaLLMService
from arena_core.parser import parse_uci_json
from finetune._common import (
    config_from_args,
    run_async_prediction_eval,
    update_eval_stats,
    write_metrics_output,
)
from finetune._common import is_legal_move as _is_legal_move
from finetune._common import rate as _rate


@dataclass(frozen=True)
class BaselineEvalConfig:
    dataset: str
    model: str
    output: str | None
    predictions_output: str | None
    limit: int | None
    ollama_base_url: str
    timeout_seconds: float
    num_ctx: int | None
    num_predict: int
    think: str


@dataclass
class BaselineEvalStats:
    examples: int = 0
    json_parse_ok: int = 0
    legal_move_ok: int = 0
    top1_match: int = 0
    service_errors: int = 0

    def to_json(self) -> dict[str, int | float]:
        return {
            "examples": self.examples,
            "json_parse_ok": self.json_parse_ok,
            "legal_move_ok": self.legal_move_ok,
            "top1_match": self.top1_match,
            "service_errors": self.service_errors,
            "json_parse_rate": _rate(self.json_parse_ok, self.examples),
            "legal_move_rate": _rate(self.legal_move_ok, self.examples),
            "top1_match_rate": _rate(self.top1_match, self.examples),
        }


def main() -> None:
    args = _parse_args()
    config = config_from_args(BaselineEvalConfig, args)
    service = OllamaLLMService(
        base_url=args.ollama_base_url,
        timeout_seconds=args.timeout_seconds,
        temperature=0.0,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        think=args.think,
    )
    stats = asyncio.run(
        evaluate_dataset(
            dataset_path=args.dataset,
            model=args.model,
            service=service,
            limit=args.limit,
            predictions_output_path=args.predictions_output,
            progress_every=args.progress_every,
        )
    )
    payload = {
        "config": asdict(config),
        "metrics": stats.to_json(),
    }
    if args.output is not None:
        write_metrics_output(args.output, payload)
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


async def evaluate_dataset(
    *,
    dataset_path: Path,
    model: str,
    service: LLMService,
    limit: int | None,
    predictions_output_path: Path | None = None,
    progress_every: int = 50,
) -> BaselineEvalStats:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    return await run_async_prediction_eval(
        dataset_path=dataset_path,
        limit=limit,
        predictions_output_path=predictions_output_path,
        stats=BaselineEvalStats(),
        predict=lambda row: evaluate_example(example=row, model=model, service=service),
        update_stats=_update_stats,
        progress_every=progress_every,
        progress=_print_progress,
    )


def _update_stats(stats: BaselineEvalStats, prediction: dict[str, Any]) -> None:
    update_eval_stats(stats, prediction)
    stats.service_errors += int(prediction["service_error"])


def _print_progress(stats: BaselineEvalStats) -> None:
    print(
        "Evaluated "
        f"{stats.examples}: parse={_rate(stats.json_parse_ok, stats.examples):.3f} "
        f"legal={_rate(stats.legal_move_ok, stats.examples):.3f} "
        f"top1={_rate(stats.top1_match, stats.examples):.3f}"
    )


async def evaluate_example(
    *,
    example: dict[str, Any],
    model: str,
    service: LLMService,
) -> dict[str, Any]:
    prompt = str(example["prompt"])
    target_move = str(example["move"])
    fen = str(example["fen"])
    try:
        response = await service.complete(model=model, prompt=prompt)
        raw_response = response.content
        service_error = None
    except Exception as exc:  # noqa: BLE001 - baseline should finish and count model failures.
        response = LLMResponse(content="")
        raw_response = ""
        service_error = f"{type(exc).__name__}: {exc}"

    parsed = parse_uci_json(raw_response)
    legal_ok = _is_legal_move(fen=fen, move_text=parsed.move) if parsed.parse_ok else False
    top1_match = parsed.move == target_move if parsed.parse_ok else False
    return {
        "fen": fen,
        "target_move": target_move,
        "raw_response": raw_response,
        "parsed_move": parsed.move,
        "parse_ok": parsed.parse_ok,
        "legal_ok": legal_ok,
        "top1_match": top1_match,
        "service_error": service_error is not None,
        "service_error_message": service_error,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base Ollama model on held-out strict-v7 JSONL examples."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Validation JSONL path.")
    parser.add_argument("--model", required=True, help="Ollama model name.")
    parser.add_argument("--output", type=Path, help="Optional JSON metrics output path.")
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Optional JSONL output with per-example predictions and parser results.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--num-ctx", type=int)
    parser.add_argument("--num-predict", type=int, default=32)
    parser.add_argument("--think", choices=["on", "off", "auto"], default="off")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()

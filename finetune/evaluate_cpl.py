from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from finetune._common import (
    config_from_args,
    run_prediction_eval,
    update_eval_stats,
    write_metrics_output,
)
from finetune._common import rate as _rate
from finetune.chess_reward import RewardConfig, StockfishRewardScorer, require_stockfish_path
from finetune.evaluate_lora import (
    _apply_chat_template,
    _decode_tokens,
    _token_id,
    _tokenize_prompt,
)


@dataclass(frozen=True)
class CPLEvalConfig:
    dataset: str
    adapter_dir: str
    output: str | None
    predictions_output: str | None
    limit: int | None
    max_seq_length: int
    max_new_tokens: int
    disable_thinking: bool
    stockfish_path: str
    reward_nodes: int
    reward_hash_mb: int
    reward_mode: str
    tactical_good_cpl: int
    tactical_inaccuracy_cpl: int
    tactical_blunder_cpl: int


@dataclass
class CPLEvalStats:
    examples: int = 0
    json_parse_ok: int = 0
    legal_move_ok: int = 0
    top1_match: int = 0
    malformed: int = 0
    illegal: int = 0
    blunders: int = 0
    mistakes: int = 0
    inaccuracies: int = 0
    reward_sum: float = 0.0
    cpl_sum: int = 0
    cpl_count: int = 0

    def to_json(self) -> dict[str, int | float | None]:
        return {
            "examples": self.examples,
            "json_parse_ok": self.json_parse_ok,
            "legal_move_ok": self.legal_move_ok,
            "top1_match": self.top1_match,
            "malformed": self.malformed,
            "illegal": self.illegal,
            "blunders": self.blunders,
            "mistakes": self.mistakes,
            "inaccuracies": self.inaccuracies,
            "json_parse_rate": _rate(self.json_parse_ok, self.examples),
            "legal_move_rate": _rate(self.legal_move_ok, self.examples),
            "top1_match_rate": _rate(self.top1_match, self.examples),
            "blunder_rate": _rate(self.blunders, self.examples),
            "mistake_rate": _rate(self.mistakes, self.examples),
            "inaccuracy_rate": _rate(self.inaccuracies, self.examples),
            "mean_reward": self.reward_sum / self.examples if self.examples else 0.0,
            "mean_generated_cpl": self.cpl_sum / self.cpl_count if self.cpl_count else None,
            "cpl_scored_legal_moves": self.cpl_count,
        }


def main() -> None:
    args = _parse_args()
    stockfish_path = require_stockfish_path(args.stockfish_path)
    config = config_from_args(
        CPLEvalConfig,
        args,
        overrides={"stockfish_path": stockfish_path},
    )
    stats = evaluate_cpl(config)
    payload = {"config": asdict(config), "metrics": stats.to_json()}
    if args.output is not None:
        write_metrics_output(args.output, payload)
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


def evaluate_cpl(config: CPLEvalConfig) -> CPLEvalStats:
    try:
        unsloth = import_module("unsloth")
        torch = import_module("torch")
    except ImportError as exc:
        raise SystemExit(
            "Evaluation dependencies are missing. Activate .venv-train and run "
            '`pip install -e ".[train]"`.'
        ) from exc

    model, tokenizer = unsloth.FastModel.from_pretrained(
        model_name=config.adapter_dir,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
    )
    model.eval()

    reward_config = RewardConfig(
        stockfish_path=config.stockfish_path,
        reward_nodes=config.reward_nodes,
        workers=1,
        hash_mb=config.reward_hash_mb,
        reward_mode=config.reward_mode,
        tactical_good_cpl=config.tactical_good_cpl,
        tactical_inaccuracy_cpl=config.tactical_inaccuracy_cpl,
        tactical_blunder_cpl=config.tactical_blunder_cpl,
    )
    with StockfishRewardScorer(reward_config) as scorer:
        return run_prediction_eval(
            dataset_path=Path(config.dataset),
            limit=config.limit,
            predictions_output_path=(
                Path(config.predictions_output) if config.predictions_output is not None else None
            ),
            stats=CPLEvalStats(),
            predict=lambda row: _evaluate_example(
                row=row,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                scorer=scorer,
                max_new_tokens=config.max_new_tokens,
                disable_thinking=config.disable_thinking,
            ),
            update_stats=_update_stats,
        )


def _evaluate_example(
    *,
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    torch: Any,
    scorer: StockfishRewardScorer,
    max_new_tokens: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    prompt = str(row["prompt"])
    target_move = str(row["move"])
    fen = str(row["fen"])
    text = _apply_chat_template(
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": prompt}],
        disable_thinking=disable_thinking,
    )
    inputs = _tokenize_prompt(tokenizer=tokenizer, text=text).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_token_id(tokenizer=tokenizer, name="pad_token_id"),
            eos_token_id=_token_id(tokenizer=tokenizer, name="eos_token_id"),
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_response = _decode_tokens(tokenizer=tokenizer, token_ids=generated_ids)
    sample = scorer.score_one(fen=fen, completion=raw_response)
    return {
        "fen": fen,
        "target_move": target_move,
        "raw_response": raw_response,
        "parsed_move": sample.parsed_move,
        "parse_ok": sample.parse_ok,
        "legal_ok": sample.legal_ok,
        "top1_match": sample.parsed_move == target_move if sample.parse_ok else False,
        "reward": sample.reward,
        "centipawn_loss": sample.centipawn_loss,
        "best_move_uci": sample.best_move_uci,
        "classification": sample.classification,
    }


def _update_stats(*, stats: CPLEvalStats, prediction: dict[str, Any]) -> None:
    update_eval_stats(stats, prediction)
    stats.malformed += int(not prediction["parse_ok"])
    stats.illegal += int(prediction["parse_ok"] and not prediction["legal_ok"])
    stats.blunders += int(prediction["classification"] == "blunder")
    stats.mistakes += int(prediction["classification"] == "mistake")
    stats.inaccuracies += int(prediction["classification"] == "inaccuracy")
    stats.reward_sum += float(prediction["reward"])
    cpl = prediction["centipawn_loss"]
    if isinstance(cpl, int):
        stats.cpl_sum += cpl
        stats.cpl_count += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated moves with Stockfish CPL on a held-out JSONL."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--stockfish-path")
    parser.add_argument("--reward-nodes", type=int, default=50_000)
    parser.add_argument("--reward-hash-mb", type=int, default=64)
    parser.add_argument("--reward-mode", choices=["linear", "tactical"], default="linear")
    parser.add_argument("--tactical-good-cpl", type=int, default=80)
    parser.add_argument("--tactical-inaccuracy-cpl", type=int, default=150)
    parser.add_argument("--tactical-blunder-cpl", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    main()

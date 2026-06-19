"""Evaluate a LoRA critic/ranker adapter on held-out candidate-level rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from finetune._common import write_metrics_output

RISK_ORDER = {"blunder": 0, "mistake": 1, "playable": 2, "good": 3}


@dataclass(frozen=True)
class CriticLoraEvalConfig:
    dataset: str
    adapter_dir: str
    output: str | None
    predictions_output: str | None
    limit: int | None
    max_seq_length: int
    max_new_tokens: int
    disable_thinking: bool


@dataclass
class CriticEvalStats:
    examples: int = 0
    json_parse_ok: int = 0
    risk_match: int = 0
    blunder_match: int = 0
    exact_target_match: int = 0
    score_abs_error_sum: int = 0

    def to_json(self) -> dict[str, int | float | None]:
        return {
            "examples": self.examples,
            "json_parse_ok": self.json_parse_ok,
            "risk_match": self.risk_match,
            "blunder_match": self.blunder_match,
            "exact_target_match": self.exact_target_match,
            "json_parse_rate": _rate(self.json_parse_ok, self.examples),
            "risk_match_rate": _rate(self.risk_match, self.examples),
            "blunder_match_rate": _rate(self.blunder_match, self.examples),
            "exact_target_match_rate": _rate(self.exact_target_match, self.examples),
            "mean_score_abs_error": _mean_error(
                self.score_abs_error_sum,
                self.json_parse_ok,
            ),
        }


def main() -> None:
    args = _parse_args()
    config = CriticLoraEvalConfig(
        dataset=str(args.dataset),
        adapter_dir=str(args.adapter_dir),
        output=str(args.output) if args.output is not None else None,
        predictions_output=(
            str(args.predictions_output) if args.predictions_output is not None else None
        ),
        limit=args.limit,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        disable_thinking=args.disable_thinking,
    )
    payload = evaluate_lora(config)
    if config.output is not None:
        write_metrics_output(Path(config.output), payload)
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


def evaluate_lora(config: CriticLoraEvalConfig) -> dict[str, Any]:
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

    stats = CriticEvalStats()
    predictions: list[dict[str, Any]] = []
    for row in _read_rows(Path(config.dataset), limit=config.limit):
        prediction = _evaluate_row(
            row=row,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            max_new_tokens=config.max_new_tokens,
            disable_thinking=config.disable_thinking,
        )
        predictions.append(prediction)
        _update_stats(stats, prediction)

    if config.predictions_output is not None:
        _write_predictions(Path(config.predictions_output), predictions)

    return {
        "config": asdict(config),
        "metrics": {
            **stats.to_json(),
            "ranking": _ranking_metrics(predictions),
        },
    }


def _evaluate_row(
    *,
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    torch: Any,
    max_new_tokens: int,
    disable_thinking: bool,
) -> dict[str, Any]:
    text = _apply_chat_template(
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": str(row["prompt"])}],
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
    parsed = parse_critic_json(raw_response)
    target = _target_from_row(row)
    return {
        "fen_before": row["fen_before"],
        "candidate_uci": row["candidate_uci"],
        "source": row["source"],
        "target": target,
        "centipawn_loss": row.get("centipawn_loss"),
        "raw_response": raw_response,
        "parse_ok": parsed is not None,
        "predicted": parsed,
        "risk_match": parsed is not None and parsed.get("risk") == target["risk"],
        "blunder_match": parsed is not None and parsed.get("blunder") == target["blunder"],
        "score_abs_error": _score_abs_error(parsed, target),
        "exact_target_match": parsed == target,
    }


def parse_critic_json(raw_response: str) -> dict[str, Any] | None:
    payload = _extract_json_object(raw_response)
    if payload is None:
        return None
    risk = payload.get("risk")
    blunder = payload.get("blunder")
    score = payload.get("score")
    reason_type = payload.get("reason_type")
    if risk not in RISK_ORDER:
        return None
    if not isinstance(blunder, bool):
        return None
    if not isinstance(score, int):
        return None
    if not isinstance(reason_type, str):
        return None
    return {
        "risk": risk,
        "blunder": blunder,
        "score": max(min(score, 100), 0),
        "reason_type": reason_type,
    }


def _extract_json_object(raw_response: str) -> dict[str, Any] | None:
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(raw_response[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _target_from_row(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("target")
    if isinstance(target, str):
        payload = json.loads(target)
    else:
        payload = json.loads(str(row["completion"]))
    return {
        "risk": str(payload["risk"]),
        "blunder": bool(payload["blunder"]),
        "score": int(payload["score"]),
        "reason_type": str(payload["reason_type"]),
    }


def _update_stats(stats: CriticEvalStats, prediction: dict[str, Any]) -> None:
    stats.examples += 1
    stats.json_parse_ok += int(prediction["parse_ok"])
    stats.risk_match += int(prediction["risk_match"])
    stats.blunder_match += int(prediction["blunder_match"])
    stats.exact_target_match += int(prediction["exact_target_match"])
    score_abs_error = prediction["score_abs_error"]
    if score_abs_error is not None:
        stats.score_abs_error_sum += int(score_abs_error)


def _ranking_metrics(predictions: list[dict[str, Any]]) -> dict[str, int | float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(prediction["fen_before"])].append(prediction)

    selected_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        parsed_rows = [row for row in rows if row["predicted"] is not None]
        if not parsed_rows:
            continue
        selected_rows.append(max(parsed_rows, key=_predicted_rank_key))
        oracle_rows.append(max(rows, key=_target_rank_key))

    selected_cpls = _cpls(selected_rows)
    oracle_cpls = _cpls(oracle_rows)
    return {
        "positions": len(selected_rows),
        "selected_mean_centipawn_loss": _mean(selected_cpls),
        "oracle_mean_centipawn_loss": _mean(oracle_cpls),
        "selected_blunders": sum(
            row["predicted"] is not None and row["predicted"].get("risk") == "blunder"
            for row in selected_rows
        ),
        "oracle_blunders": sum(row["target"]["risk"] == "blunder" for row in oracle_rows),
        "oracle_match_by_move": sum(
            selected["candidate_uci"] == oracle["candidate_uci"]
            for selected, oracle in zip(selected_rows, oracle_rows, strict=True)
        ),
    }


def _predicted_rank_key(row: dict[str, Any]) -> tuple[int, int]:
    predicted = row["predicted"]
    return int(predicted["score"]), RISK_ORDER[str(predicted["risk"])]


def _target_rank_key(row: dict[str, Any]) -> tuple[int, int]:
    target = row["target"]
    return int(target["score"]), RISK_ORDER[str(target["risk"])]


def _score_abs_error(parsed: dict[str, Any] | None, target: dict[str, Any]) -> int | None:
    if parsed is None:
        return None
    return abs(int(parsed["score"]) - int(target["score"]))


def _cpls(rows: list[dict[str, Any]]) -> list[int]:
    result: list[int] = []
    for row in rows:
        cpl = row.get("centipawn_loss")
        if isinstance(cpl, int):
            result.append(cpl)
    return result


def _read_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, separators=(",", ":")) + "\n")


def _apply_chat_template(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    disable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        try:
            return str(
                tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **kwargs,
                )
            )
        except TypeError:
            pass
    return str(tokenizer.apply_chat_template(messages, **kwargs))


def _tokenize_prompt(*, tokenizer: Any, text: str) -> Any:
    try:
        return tokenizer(text, return_tensors="pt")
    except (TypeError, ValueError):
        return tokenizer(text=text, return_tensors="pt")


def _decode_tokens(*, tokenizer: Any, token_ids: Any) -> str:
    decoder = getattr(tokenizer, "decode", None)
    if decoder is None:
        decoder = tokenizer.tokenizer.decode
    return str(decoder(token_ids, skip_special_tokens=False))


def _token_id(*, tokenizer: Any, name: str) -> int | None:
    value = getattr(tokenizer, name, None)
    if value is not None:
        return int(value)
    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if inner_tokenizer is None:
        return None
    inner_value = getattr(inner_tokenizer, name, None)
    return int(inner_value) if inner_value is not None else None


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 4)


def _mean_error(total_error: int, count: int) -> float | None:
    if count == 0:
        return None
    return round(total_error / count, 2)


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA critic/ranker adapter.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

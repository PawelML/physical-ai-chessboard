"""Evaluate a critic/ranker LoRA by risk-label log probabilities.

This avoids slow JSON generation. For each candidate row, the model is asked only how
likely the next risk label is after the shared JSON prefix `{"risk":"`.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO

from finetune._common import config_from_args, read_jsonl, write_metrics_output
from finetune.evaluate_critic_ranker_lora import RISK_ORDER
from finetune.evaluate_lora import _apply_chat_template

RISK_LABEL_PREFIX = '{"risk":"'
RISK_SCORE = {"blunder": 0.0, "mistake": 1.0, "playable": 2.0, "good": 3.0}


@dataclass(frozen=True)
class RiskLogprobEvalConfig:
    dataset: str
    adapter_dir: str
    output: str | None
    predictions_output: str | None
    limit: int | None
    max_seq_length: int
    disable_thinking: bool


def main() -> None:
    args = _parse_args()
    config = config_from_args(RiskLogprobEvalConfig, args)
    payload = evaluate_risk_logprobs(config)
    if config.output is not None:
        write_metrics_output(Path(config.output), payload)
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


def evaluate_risk_logprobs(config: RiskLogprobEvalConfig) -> dict[str, Any]:
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

    predictions: list[dict[str, Any]] = []
    predictions_file = _open_optional_predictions(
        Path(config.predictions_output) if config.predictions_output is not None else None
    )
    try:
        for index, row in enumerate(read_jsonl(Path(config.dataset))):
            if config.limit is not None and index >= config.limit:
                break
            prediction = _evaluate_row(
                row=row,
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                disable_thinking=config.disable_thinking,
            )
            predictions.append(prediction)
            _write_prediction(predictions_file, prediction)
    finally:
        if predictions_file is not None:
            predictions_file.close()

    return {
        "config": asdict(config),
        "metrics": {
            **_classification_metrics(predictions),
            "ranking": ranking_metrics(predictions),
        },
    }


def _evaluate_row(
    *,
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    torch: Any,
    disable_thinking: bool,
) -> dict[str, Any]:
    prompt_text = _apply_chat_template(
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": str(row["prompt"])}],
        disable_thinking=disable_thinking,
    )
    prefix_text = prompt_text + RISK_LABEL_PREFIX
    label_logprobs = _label_logprobs(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        prefix_text=prefix_text,
        labels=list(RISK_ORDER),
    )
    risk_probs = softmax(label_logprobs)
    predicted_risk = max(label_logprobs, key=label_logprobs.get)
    expected_risk_score = sum(
        RISK_SCORE[risk] * probability
        for risk, probability in risk_probs.items()
    )
    target = _target_from_row(row)
    return {
        "fen_before": row["fen_before"],
        "candidate_uci": row["candidate_uci"],
        "candidate_rank_in_generator": row.get("candidate_rank_in_generator"),
        "source": row.get("source"),
        "target": target,
        "centipawn_loss": row.get("centipawn_loss"),
        "risk_logprobs": label_logprobs,
        "risk_probs": risk_probs,
        "predicted_risk": predicted_risk,
        "expected_risk_score": expected_risk_score,
        "risk_match": predicted_risk == target["risk"],
        "blunder_match": (predicted_risk == "blunder") == bool(target["blunder"]),
    }


def _label_logprobs(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    prefix_text: str,
    labels: list[str],
) -> dict[str, float]:
    sequences: list[list[int]] = []
    label_starts: list[int] = []
    prefix_ids = _token_ids(tokenizer=tokenizer, text=prefix_text)
    for label in labels:
        full_ids = _token_ids(tokenizer=tokenizer, text=prefix_text + label)
        label_start = _common_prefix_length(prefix_ids, full_ids)
        if label_start >= len(full_ids):
            raise ValueError(f"risk label {label!r} produced no label tokens")
        sequences.append(full_ids)
        label_starts.append(label_start)

    input_ids, attention_mask = _padded_batch(
        torch=torch,
        tokenizer=tokenizer,
        sequences=sequences,
        device=model.device,
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return {
        label: _sequence_label_logprob(
            log_probs=log_probs[row_index],
            token_ids=sequences[row_index],
            label_start=label_starts[row_index],
        )
        for row_index, label in enumerate(labels)
    }


def _padded_batch(
    *,
    torch: Any,
    tokenizer: Any,
    sequences: list[list[int]],
    device: Any,
) -> tuple[Any, Any]:
    pad_token_id = _token_id(tokenizer=tokenizer, name="pad_token_id")
    if pad_token_id is None:
        pad_token_id = _token_id(tokenizer=tokenizer, name="eos_token_id")
    if pad_token_id is None:
        pad_token_id = 0
    max_length = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [pad_token_id] * (max_length - len(sequence))
        for sequence in sequences
    ]
    mask = [
        [1] * len(sequence) + [0] * (max_length - len(sequence))
        for sequence in sequences
    ]
    return (
        torch.tensor(padded, device=device),
        torch.tensor(mask, device=device),
    )


def _sequence_label_logprob(
    *,
    log_probs: Any,
    token_ids: list[int],
    label_start: int,
) -> float:
    total = 0.0
    token_count = 0
    for token_index in range(label_start, len(token_ids)):
        previous_index = token_index - 1
        if previous_index < 0:
            continue
        total += float(log_probs[previous_index, token_ids[token_index]].item())
        token_count += 1
    if token_count == 0:
        raise ValueError("risk label has no scorable tokens")
    return total / token_count


def _token_ids(*, tokenizer: Any, text: str) -> list[int]:
    try:
        tokens = tokenizer(text, add_special_tokens=False, return_tensors=None)
    except (TypeError, ValueError):
        tokens = tokenizer(text=text, add_special_tokens=False, return_tensors=None)
    ids = tokens["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    index = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        index += 1
    return index


def softmax(logprobs: dict[str, float]) -> dict[str, float]:
    max_logprob = max(logprobs.values())
    exp_values = {
        label: math.exp(logprob - max_logprob)
        for label, logprob in logprobs.items()
    }
    total = sum(exp_values.values())
    return {label: value / total for label, value in exp_values.items()}


def ranking_metrics(predictions: list[dict[str, Any]]) -> dict[str, int | float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(prediction["fen_before"])].append(prediction)

    selected_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    for rows in grouped.values():
        selected_rows.append(max(rows, key=_predicted_rank_key))
        oracle_rows.append(max(rows, key=_target_rank_key))
        first_rows.append(min(rows, key=_generator_rank_key))

    selected_cpls = _cpls(selected_rows)
    oracle_cpls = _cpls(oracle_rows)
    first_cpls = _cpls(first_rows)
    return {
        "positions": len(selected_rows),
        "selected_mean_centipawn_loss": _mean(selected_cpls),
        "oracle_mean_centipawn_loss": _mean(oracle_cpls),
        "first_generator_mean_centipawn_loss": _mean(first_cpls),
        "selected_blunders": _risk_count(selected_rows, "blunder"),
        "oracle_blunders": _risk_count(oracle_rows, "blunder"),
        "first_generator_blunders": _risk_count(first_rows, "blunder"),
        "oracle_match_by_move": sum(
            selected["candidate_uci"] == oracle["candidate_uci"]
            for selected, oracle in zip(selected_rows, oracle_rows, strict=True)
        ),
        "improved_vs_first_count": _compare_cpl(selected_rows, first_rows, operator="<"),
        "worsened_vs_first_count": _compare_cpl(selected_rows, first_rows, operator=">"),
        "tied_first_count": _compare_cpl(selected_rows, first_rows, operator="=="),
    }


def _classification_metrics(predictions: list[dict[str, Any]]) -> dict[str, int | float]:
    examples = len(predictions)
    risk_match = sum(bool(prediction["risk_match"]) for prediction in predictions)
    blunder_match = sum(bool(prediction["blunder_match"]) for prediction in predictions)
    return {
        "examples": examples,
        "risk_match": risk_match,
        "risk_match_rate": risk_match / examples if examples else 0.0,
        "blunder_match": blunder_match,
        "blunder_match_rate": blunder_match / examples if examples else 0.0,
    }


def _predicted_rank_key(row: dict[str, Any]) -> tuple[float, int]:
    return float(row["expected_risk_score"]), -_generator_rank_value(row)


def _target_rank_key(row: dict[str, Any]) -> tuple[int, int, int]:
    target = row["target"]
    cpl = row.get("centipawn_loss")
    cpl_key = -999_999 if not isinstance(cpl, int) else -cpl
    return int(target["score"]), RISK_ORDER[str(target["risk"])], cpl_key


def _generator_rank_key(row: dict[str, Any]) -> int:
    return _generator_rank_value(row)


def _generator_rank_value(row: dict[str, Any]) -> int:
    value = row.get("candidate_rank_in_generator")
    return int(value) if isinstance(value, int) else 999_999


def _target_from_row(row: dict[str, Any]) -> dict[str, Any]:
    target = row.get("target")
    payload = json.loads(target) if isinstance(target, str) else json.loads(str(row["completion"]))
    return {
        "risk": str(payload["risk"]),
        "blunder": bool(payload["blunder"]),
        "score": int(payload["score"]),
        "reason_type": str(payload["reason_type"]),
    }


def _cpls(rows: list[dict[str, Any]]) -> list[int]:
    return [
        int(row["centipawn_loss"])
        for row in rows
        if isinstance(row.get("centipawn_loss"), int)
    ]


def _risk_count(rows: list[dict[str, Any]], risk: str) -> int:
    return sum(str(row["target"]["risk"]) == risk for row in rows)


def _compare_cpl(
    selected_rows: list[dict[str, Any]],
    first_rows: list[dict[str, Any]],
    *,
    operator: str,
) -> int:
    count = 0
    for selected, first in zip(selected_rows, first_rows, strict=True):
        selected_cpl = selected.get("centipawn_loss")
        first_cpl = first.get("centipawn_loss")
        if not isinstance(selected_cpl, int) or not isinstance(first_cpl, int):
            continue
        if operator == "<" and selected_cpl < first_cpl:
            count += 1
        elif operator == ">" and selected_cpl > first_cpl:
            count += 1
        elif operator == "==" and selected_cpl == first_cpl:
            count += 1
    return count


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _token_id(*, tokenizer: Any, name: str) -> int | None:
    value = getattr(tokenizer, name, None)
    if value is not None:
        return int(value)
    inner_tokenizer = getattr(tokenizer, "tokenizer", None)
    if inner_tokenizer is None:
        return None
    inner_value = getattr(inner_tokenizer, name, None)
    return int(inner_value) if inner_value is not None else None


def _open_optional_predictions(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _write_prediction(output_file: TextIO | None, prediction: dict[str, Any]) -> None:
    if output_file is not None:
        output_file.write(json.dumps(prediction, separators=(",", ":")) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate critic risk labels with log probabilities instead of generation."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

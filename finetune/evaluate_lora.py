from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TextIO

import chess

from arena_core.parser import parse_uci_json


@dataclass(frozen=True)
class LoraEvalConfig:
    dataset: str
    adapter_dir: str
    output: str | None
    predictions_output: str | None
    limit: int | None
    max_seq_length: int
    max_new_tokens: int
    disable_thinking: bool


@dataclass
class LoraEvalStats:
    examples: int = 0
    json_parse_ok: int = 0
    legal_move_ok: int = 0
    top1_match: int = 0

    def to_json(self) -> dict[str, int | float]:
        return {
            "examples": self.examples,
            "json_parse_ok": self.json_parse_ok,
            "legal_move_ok": self.legal_move_ok,
            "top1_match": self.top1_match,
            "json_parse_rate": _rate(self.json_parse_ok, self.examples),
            "legal_move_rate": _rate(self.legal_move_ok, self.examples),
            "top1_match_rate": _rate(self.top1_match, self.examples),
        }


def main() -> None:
    args = _parse_args()
    config = LoraEvalConfig(
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
    stats = evaluate_lora(config)
    payload = {
        "config": asdict(config),
        "metrics": stats.to_json(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


def evaluate_lora(config: LoraEvalConfig) -> LoraEvalStats:
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

    predictions_file: TextIO | None = None
    stats = LoraEvalStats()
    try:
        if config.predictions_output is not None:
            predictions_path = Path(config.predictions_output)
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_file = predictions_path.open("w", encoding="utf-8")

        with Path(config.dataset).open(encoding="utf-8") as dataset_file:
            for line in dataset_file:
                if config.limit is not None and stats.examples >= config.limit:
                    break
                row = json.loads(line)
                prediction = _evaluate_example(
                    row=row,
                    model=model,
                    tokenizer=tokenizer,
                    torch=torch,
                    max_new_tokens=config.max_new_tokens,
                    disable_thinking=config.disable_thinking,
                )
                stats.examples += 1
                stats.json_parse_ok += int(prediction["parse_ok"])
                stats.legal_move_ok += int(prediction["legal_ok"])
                stats.top1_match += int(prediction["top1_match"])
                if predictions_file is not None:
                    predictions_file.write(json.dumps(prediction, separators=(",", ":")) + "\n")
    finally:
        if predictions_file is not None:
            predictions_file.close()
    return stats


def _evaluate_example(
    *,
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    torch: Any,
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
    parsed = parse_uci_json(raw_response)
    legal_ok = _is_legal_move(fen=fen, move_text=parsed.move) if parsed.parse_ok else False
    return {
        "fen": fen,
        "target_move": target_move,
        "raw_response": raw_response,
        "parsed_move": parsed.move,
        "parse_ok": parsed.parse_ok,
        "legal_ok": legal_ok,
        "top1_match": parsed.move == target_move if parsed.parse_ok else False,
    }


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
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


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


def _is_legal_move(*, fen: str, move_text: str | None) -> bool:
    if move_text is None:
        return False
    try:
        move = chess.Move.from_uci(move_text)
    except ValueError:
        return False
    return move in chess.Board(fen).legal_moves


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter with HF/Unsloth.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "unsloth/Qwen3.5-9B"
DEFAULT_INSTRUCTION_PART = "<|im_start|>user\n"
DEFAULT_RESPONSE_PART = "<|im_start|>assistant\n"


@dataclass(frozen=True)
class TrainConfig:
    train_dataset: str
    eval_dataset: str | None
    output_dir: str
    model: str
    max_seq_length: int
    max_steps: int
    num_train_epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    logging_steps: int
    eval_steps: int | None
    save_steps: int
    save_total_limit: int
    lora_rank: int
    lora_alpha: int
    seed: int
    instruction_part: str
    response_part: str


def main() -> None:
    args = _parse_args()
    config = TrainConfig(
        train_dataset=str(args.train_dataset),
        eval_dataset=str(args.eval_dataset) if args.eval_dataset is not None else None,
        output_dir=str(args.output_dir),
        model=args.model,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
        instruction_part=args.instruction_part,
        response_part=args.response_part,
    )
    train(config)


def train(config: TrainConfig) -> None:
    try:
        unsloth = import_module("unsloth")
        chat_templates = import_module("unsloth.chat_templates")
        datasets = import_module("datasets")
        trl = import_module("trl")
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Activate .venv-train and run "
            '`pip install -e ".[train]"`.'
        ) from exc

    FastModel = unsloth.FastModel
    train_on_responses_only = chat_templates.train_on_responses_only
    load_dataset = datasets.load_dataset
    SFTConfig = trl.SFTConfig
    SFTTrainer = trl.SFTTrainer

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    model = FastModel.get_peft_model(
        model,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )

    train_dataset = _load_and_format_dataset(
        load_dataset=load_dataset,
        tokenizer=tokenizer,
        dataset_path=Path(config.train_dataset),
    )
    eval_dataset = (
        _load_and_format_dataset(
            load_dataset=load_dataset,
            tokenizer=tokenizer,
            dataset_path=Path(config.eval_dataset),
        )
        if config.eval_dataset is not None
        else None
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        dataset_text_field="text",
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset is not None and config.eval_steps else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=config.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part=config.instruction_part,
        response_part=config.response_part,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


def _load_and_format_dataset(
    *,
    load_dataset: Any,
    tokenizer: Any,
    dataset_path: Path,
) -> Any:
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    def format_examples(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        texts = []
        for prompt, completion in zip(batch["prompt"], batch["completion"], strict=True):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return {"text": texts}

    return dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 2 QLoRA/LoRA fine-tuning.")
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--eval-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-part", default=DEFAULT_INSTRUCTION_PART)
    parser.add_argument("--response-part", default=DEFAULT_RESPONSE_PART)
    return parser.parse_args()


if __name__ == "__main__":
    main()

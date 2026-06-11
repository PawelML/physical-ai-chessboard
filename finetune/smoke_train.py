from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

DEFAULT_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"


def main() -> None:
    args = _parse_args()
    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


def train(
    *,
    dataset_path: Path,
    output_dir: Path,
    model_name: str,
    max_seq_length: int,
    max_steps: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
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
    FastLanguageModel = unsloth.FastLanguageModel
    get_chat_template = chat_templates.get_chat_template
    train_on_responses_only = chat_templates.train_on_responses_only
    load_dataset = datasets.load_dataset
    SFTConfig = trl.SFTConfig
    SFTTrainer = trl.SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

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

    dataset = dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field="text",
            max_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=0,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Qwen2.5 Phase 0 LoRA smoke train.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main()

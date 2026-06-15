from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from finetune._common import config_from_args, write_metrics_output
from finetune.chess_reward import RewardConfig, StockfishRewardScorer, require_stockfish_path
from finetune.evaluate_lora import _apply_chat_template

DEFAULT_MODEL = "unsloth/Qwen3.5-9B"
DEFAULT_DISTILLED_ADAPTER_DIR = Path(
    "outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora"
)
DEFAULT_MERGED_MODEL_DIR = Path(
    "outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf"
)
DEFAULT_TRAIN_DATASET = Path(
    "data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl"
)


@dataclass(frozen=True)
class GRPOTrainConfig:
    train_dataset: str
    output_dir: str
    model: str
    distilled_adapter_dir: str
    merged_model_dir: str
    initial_adapter_dir: str | None
    auto_merge_distilled: bool
    max_seq_length: int
    max_prompt_length: int
    max_completion_length: int
    max_steps: int
    num_train_epochs: float
    learning_rate: float
    beta: float
    num_generations: int
    temperature: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    lora_rank: int
    lora_alpha: int
    seed: int
    limit: int | None
    reward_nodes: int
    reward_workers: int
    reward_hash_mb: int
    reward_mode: str
    tactical_good_cpl: int
    tactical_inaccuracy_cpl: int
    tactical_blunder_cpl: int
    stockfish_path: str


def main() -> None:
    args = _parse_args()
    stockfish_path = require_stockfish_path(args.stockfish_path)
    config = config_from_args(
        GRPOTrainConfig,
        args,
        overrides={
            "auto_merge_distilled": not args.no_auto_merge_distilled,
            "stockfish_path": stockfish_path,
        },
    )
    train(config)


def train(config: GRPOTrainConfig) -> None:
    try:
        unsloth = import_module("unsloth")
        datasets = import_module("datasets")
        trl = import_module("trl")
        _normalize_trl_optional_dependency_flags(import_module("trl.import_utils"))
        GRPOConfig = trl.GRPOConfig
        GRPOTrainer = trl.GRPOTrainer
    except (ImportError, RuntimeError) as exc:
        raise SystemExit(
            "GRPO dependencies are missing or incompatible. Activate .venv-train and run "
            '`pip install -e ".[train]"`. If this mentions vLLM, install a TRL/vLLM '
            "combination where GRPOTrainer imports cleanly even with use_vllm=False."
        ) from exc

    FastModel = unsloth.FastModel
    load_dataset = datasets.load_dataset

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_output(output_dir / "grpo_config.json", asdict(config))

    merged_model_dir = Path(config.merged_model_dir)
    if not _looks_like_hf_checkpoint(merged_model_dir):
        if not config.auto_merge_distilled:
            raise SystemExit(
                f"Merged distilled checkpoint is missing at {merged_model_dir}; "
                "rerun without --no-auto-merge-distilled or create it manually."
            )
        _merge_distilled_adapter(
            FastModel=FastModel,
            adapter_dir=Path(config.distilled_adapter_dir),
            output_dir=merged_model_dir,
            max_seq_length=config.max_seq_length,
        )

    if config.initial_adapter_dir is not None:
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.initial_adapter_dir,
            max_seq_length=config.max_seq_length,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        if hasattr(FastModel, "for_training"):
            model = FastModel.for_training(model)
    else:
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(merged_model_dir),
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
    trainable_params = _count_trainable_parameters(model)
    if trainable_params == 0:
        raise SystemExit("Loaded model has no trainable parameters; cannot run GRPO.")

    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = _load_grpo_dataset(
        load_dataset=load_dataset,
        tokenizer=tokenizer,
        dataset_path=Path(config.train_dataset),
        limit=config.limit,
    )
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        num_generations=config.num_generations,
        temperature=config.temperature,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        seed=config.seed,
        report_to="none",
        use_vllm=False,
        remove_unused_columns=False,
    )
    reward_config = RewardConfig(
        stockfish_path=config.stockfish_path,
        reward_nodes=config.reward_nodes,
        workers=config.reward_workers,
        hash_mb=config.reward_hash_mb,
        reward_mode=config.reward_mode,
        tactical_good_cpl=config.tactical_good_cpl,
        tactical_inaccuracy_cpl=config.tactical_inaccuracy_cpl,
        tactical_blunder_cpl=config.tactical_blunder_cpl,
    )
    with StockfishRewardScorer(
        reward_config,
        metrics_output=output_dir / "reward_metrics.jsonl",
    ) as reward_scorer:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_scorer,
            args=grpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    print(f"Saved GRPO LoRA adapter to {output_dir}")


def _normalize_trl_optional_dependency_flags(trl_import_utils: Any) -> None:
    """Handle Transformers 5.x tuple returns cached by TRL as optional-dependency flags."""
    for name, value in vars(trl_import_utils).items():
        if name.endswith("_available") and isinstance(value, tuple):
            setattr(trl_import_utils, name, bool(value[0]))


def _merge_distilled_adapter(
    *,
    FastModel: Any,
    adapter_dir: Path,
    output_dir: Path,
    max_seq_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method="merged_16bit",
    )


def _load_grpo_dataset(
    *,
    load_dataset: Any,
    tokenizer: Any,
    dataset_path: Path,
    limit: int | None,
) -> Any:
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    def format_examples(batch: dict[str, list[str]]) -> dict[str, list[str]]:
        prompts = []
        for prompt in batch["prompt"]:
            prompts.append(
                _apply_chat_template(
                    tokenizer=tokenizer,
                    messages=[{"role": "user", "content": prompt}],
                    disable_thinking=True,
                )
            )
        return {"prompt": prompts, "fen": batch["fen"]}

    return dataset.map(format_examples, batched=True, remove_columns=dataset.column_names)


def _looks_like_hf_checkpoint(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists()


def _count_trainable_parameters(model: Any) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO with a Stockfish CPL reward.")
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--distilled-adapter-dir", type=Path, default=DEFAULT_DISTILLED_ADAPTER_DIR)
    parser.add_argument("--merged-model-dir", type=Path, default=DEFAULT_MERGED_MODEL_DIR)
    parser.add_argument(
        "--initial-adapter-dir",
        type=Path,
        help="Existing LoRA adapter to continue training from instead of creating a fresh LoRA.",
    )
    parser.add_argument("--no-auto-merge-distilled", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--max-completion-length", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--reward-nodes", type=int, default=50_000)
    parser.add_argument("--reward-mode", choices=["linear", "tactical"], default="linear")
    parser.add_argument("--tactical-good-cpl", type=int, default=80)
    parser.add_argument("--tactical-inaccuracy-cpl", type=int, default=150)
    parser.add_argument("--tactical-blunder-cpl", type=int, default=300)
    parser.add_argument(
        "--reward-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
    )
    parser.add_argument("--reward-hash-mb", type=int, default=64)
    parser.add_argument("--stockfish-path")
    return parser.parse_args()


if __name__ == "__main__":
    main()

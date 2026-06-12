# Phase 0 Fine-Tuning Smoke Test

This directory contains the local smoke pipeline for the fine-tuning plan. It is
deliberately separate from the normal arena environment because Unsloth pulls a
large training stack: Torch, Transformers, PEFT, TRL, bitsandbytes, and friends.

## Environment

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install --upgrade pip
pip install -e ".[train]"
```

On this machine the resolved stack is pinned in `pyproject.toml` to avoid slow
resolver backtracking. With Python 3.12, xFormers may warn that its C++/CUDA
extensions do not match the installed Torch build; Unsloth falls back to PyTorch
attention. That is slower but fine for Phase 0.

The Phase 0 model is:

- Ollama runtime model: `qwen2.5:1.5b`
- Unsloth/Hugging Face training model:
  `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

Pull the Ollama baseline:

```bash
ollama pull qwen2.5:1.5b
```

## Build Smoke Examples

Use any small PGN file first. Generated data belongs under `data/finetune/`,
which is ignored by git.

```bash
python -m finetune.build_smoke_dataset \
  --pgn path/to/small.pgn \
  --output data/finetune/qwen25_15b_smoke.jsonl \
  --metadata-output data/finetune/qwen25_15b_smoke.meta.json \
  --max-examples 2000 \
  --seed 0
```

The examples are rendered with `arena_core.prompts.build_strict_prompt`; the
training target is exactly `{"move":"<uci>"}`.

## Build Phase 1 Dataset

For real Lichess dumps, stream compressed PGN into the builder; do not extract
monthly dumps to disk.

```bash
zstdcat path/to/lichess_db_standard_rated_YYYY-MM.pgn.zst \
  | python scripts/build_finetune_dataset.py \
      --pgn - \
      --train-output data/finetune/lichess_2000_pilot.train.jsonl \
      --val-output data/finetune/lichess_2000_pilot.val.jsonl \
      --metadata-output data/finetune/lichess_2000_pilot.meta.json \
      --max-examples 20000 \
      --seed 0
```

Defaults match the Phase 1 plan:

- rated games only;
- blitz or slower;
- both players at least 2000 Elo;
- normal termination;
- at least 20 plies;
- up to 10 evenly-spaced positions per kept game;
- train/val split by game, not by position;
- 70% open prompts and 30% constrained prompts.

Use `--max-examples 200000` for the first main dataset after the 20k pilot
looks sane.

## Evaluate Base Model on Held-Out Examples

Before training, record the base model's parse/legal/top-1 metrics on the
validation split.

```bash
python -m finetune.evaluate_baseline \
  --dataset data/finetune/lichess_2000_pilot.val.jsonl \
  --model qwen2.5:1.5b \
  --output outputs/finetune/qwen25_15b_base_pilot_eval.json \
  --predictions-output outputs/finetune/qwen25_15b_base_pilot_predictions.jsonl \
  --limit 1000
```

## Train Tiny LoRA

This is intentionally short. It is meant to catch CUDA, tokenizer, chat-template,
and adapter-save issues before using a real dataset.

```bash
python -m finetune.smoke_train \
  --dataset data/finetune/qwen25_15b_smoke.jsonl \
  --output-dir outputs/finetune/qwen25_15b_smoke_lora \
  --max-steps 100
```

After this succeeds, merge/convert to GGUF with the current Unsloth and
llama.cpp workflow, then create an Ollama model with the base model's template.
Keep generated adapters, merged weights, and GGUF files under
`outputs/finetune/`.

## Phase 2 Qwen3.5 9B Pilot Training

Start with the 20k pilot dataset before training on the 200k main dataset.

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
python -m finetune.train_lora \
  --model unsloth/Qwen3.5-9B \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot.train.jsonl \
  --eval-dataset data/finetune/lichess_2000_2013-12_pilot.val.jsonl \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_lora \
  --max-seq-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4
```

The script stores `train_config.json` next to the LoRA adapter. It uses the
model tokenizer's own chat template and masks loss to assistant responses only.

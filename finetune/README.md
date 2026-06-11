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

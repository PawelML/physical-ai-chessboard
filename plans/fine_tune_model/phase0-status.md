# Phase 0 Status - Environment + Smoke Test

Status: completed
Date: 2026-06-11

## Summary

Phase 0 of `finetuning-chess-llm-plan.md` is complete for the chosen smoke-test
model pair:

- Ollama runtime model: `qwen2.5:1.5b`
- Unsloth/Hugging Face training model:
  `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`

The local end-to-end smoke path has been proven through:

1. isolated training environment creation;
2. Ollama model pull;
3. PGN to strict-v7 JSONL dataset generation;
4. one-step QLoRA/LoRA training probe on the RTX 3090;
5. adapter save under ignored fine-tuning outputs.

## Implemented Repo Changes

Added fine-tuning utilities:

- `finetune/build_smoke_dataset.py`
  - Reads PGN.
  - Replays games with `python-chess`.
  - Renders examples using the real `arena_core.prompts.build_strict_prompt`.
  - Writes JSONL with `prompt`, `completion`, UCI move, SAN, FEN, prompt version,
    prompt template hash, legality mode, and source metadata.
- `finetune/smoke_train.py`
  - Runs a small Unsloth SFT/QLoRA training job.
  - Defaults to `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`.
  - Saves LoRA adapters to the requested output directory.
- `finetune/README.md`
  - Contains the setup and Phase 0 commands.
- `tests/test_finetune_smoke_dataset.py`
  - Verifies strict-v7 prompt generation and UCI JSON completion targets.

Updated config:

- `.gitignore`
  - Ignores `.venv-train/`, `data/finetune/`, `outputs/finetune/`,
    `unsloth_compiled_cache/`, `*.safetensors`, and `*.gguf`.
- `pyproject.toml`
  - Adds `finetune` to package build targets.
  - Adds pinned `train` extras:
    - `datasets==5.0.0`
    - `unsloth==2025.11.1`

## Local Environment State

Created local training venv:

```bash
.venv-train
```

Installed training dependencies with:

```bash
.venv-train/bin/python -m pip install -e ".[train]"
```

Pulled Ollama model:

```bash
ollama pull qwen2.5:1.5b
```

Confirmed installed Ollama model:

```text
qwen2.5:1.5b    65ec06548149    986 MB
```

Generated local smoke artifacts:

- `data/finetune/phase0_smoke.pgn`
- `data/finetune/qwen25_15b_smoke.jsonl`
- `data/finetune/qwen25_15b_smoke.meta.json`
- `outputs/finetune/qwen25_15b_probe_lora/`

These are intentionally ignored by git.

Approximate ignored local sizes after Phase 0:

```text
.venv-train       5.5G
data/finetune      24K
outputs/finetune  208M
```

## Validation Performed

Code validation:

```bash
.venv/bin/python -m ruff check finetune tests/test_finetune_smoke_dataset.py pyproject.toml
.venv/bin/python -m pytest tests/test_finetune_smoke_dataset.py tests/test_prompts.py
```

Result:

```text
All checks passed.
3 passed.
```

Training environment import check:

```bash
.venv-train/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
import unsloth
print("unsloth", getattr(unsloth, "__version__", "unknown"))
PY
```

Result:

```text
torch 2.12.0+cu130
cuda_available True
cuda_device NVIDIA GeForce RTX 3090
unsloth 2025.11.1
```

Smoke dataset generation command that succeeded:

```bash
.venv-train/bin/python -m finetune.build_smoke_dataset \
  --pgn data/finetune/phase0_smoke.pgn \
  --output data/finetune/qwen25_15b_smoke.jsonl \
  --metadata-output data/finetune/qwen25_15b_smoke.meta.json \
  --max-examples 6 \
  --seed 0
```

One-step training probe that succeeded:

```bash
.venv-train/bin/python -m finetune.smoke_train \
  --dataset data/finetune/qwen25_15b_smoke.jsonl \
  --output-dir outputs/finetune/qwen25_15b_probe_lora \
  --max-steps 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 1
```

Training probe result:

```text
Num examples = 6
Total steps = 1
Trainable parameters = 18,464,768 of 1,562,179,072
train_loss = 1.0579942464828491
Saved LoRA adapter to outputs/finetune/qwen25_15b_probe_lora
```

## Known Issue: xFormers Fallback

xFormers currently warns that its compiled extensions do not match the installed
Torch/Python stack:

```text
xFormers built for: PyTorch 2.10.0+cu128, Python 3.10.19
Current stack:      PyTorch 2.12.0+cu130, Python 3.12.3
```

Current conclusion:

- Do not block Phase 0 on this.
- Unsloth falls back to PyTorch attention and the one-step training probe still
  succeeds on the RTX 3090.
- Revisit only if the 20k-example pilot is too slow or VRAM-constrained.

Recommended future fix, if needed:

- Keep repo/arena/dataset generation on Python 3.12.
- Create a separate optimized training env, likely Python 3.10 or 3.11, aligned
  with an xFormers/Torch wheel set.
- Do not mutate the current working `.venv-train` unless there is a measured
  reason.

## Next Recommended Step

Proceed to Phase 1 dataset work:

1. Choose a small real PGN source first.
2. Use `finetune/build_smoke_dataset.py` as the starting point.
3. Expand it toward the plan's full `scripts/build_finetune_dataset.py` behavior:
   filtering, train/val split by game, legality-mode mix, sidecar provenance,
   and larger example counts.
4. Before real training, add the plan's pre-training baseline evaluation over
   held-out examples: JSON parse rate, legal-move rate, and top-1 match rate.

Important invariant:

- Training examples must continue to be rendered by
  `arena_core.prompts.build_strict_prompt`.
- Completion targets must remain exactly `{"move":"<uci>"}`.
- Do not change the benchmark prompt contract to make the fine-tune look better.

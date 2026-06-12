# Phase 2 Status - Train on the RTX 3090

Status: 20k pilot training completed
Date: 2026-06-11

## Model Choice

Chosen first real model:

- Ollama baseline/runtime family: `qwen3.5:9b`
- Hugging Face/Unsloth training checkpoint: `unsloth/Qwen3.5-9B`

Reasoning:

- recent Qwen3.5 model family;
- practical size for QLoRA on a 24 GB RTX 3090;
- local Ollama baseline already exists as `qwen3.5:9b`;
- Unsloth supports Qwen3.5 fine-tuning;
- Apache-2.0 license surfaced by the local Ollama model metadata.

## Implemented Repo Changes

Added:

- `finetune/train_lora.py`
  - Phase 2 QLoRA/LoRA training entrypoint.
  - Defaults to `unsloth/Qwen3.5-9B`.
  - Uses `FastModel`.
  - Uses the model tokenizer's own chat template.
  - Masks loss to assistant responses only via Unsloth's
    `train_on_responses_only`.
  - Writes `train_config.json` into the output directory before training.

Updated:

- `finetune/README.md`
  - Adds the Qwen3.5 9B pilot training command.
- `pyproject.toml`
  - Updates train extras to `unsloth==2026.6.2`, `transformers==5.5.0`,
    and `datasets==4.3.0` for Qwen3.5 support.

## Local Training Stack

The initial probe failed with `transformers==4.57.2` because that version did
not know model type `qwen3_5`. The training venv was upgraded to:

```text
torch 2.10.0+cu128
transformers 5.5.0
datasets 4.3.0
unsloth 2026.6.2
cuda device NVIDIA GeForce RTX 3090
```

Hugging Face/Xet fast download stalled for `unsloth/Qwen3.5-9B`; downloading
with `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0` completed successfully.
The model snapshot is cached locally under Hugging Face cache and occupies about
19 GB.

## Sequence Length Check

Tokenized pilot dataset with Qwen3.5 chat template:

```text
count = 20000
mean = 654.7
p50 = 630
p90 = 945
p95 = 1033
p99 = 1234
p99.5 = 1334
p99.9 = 1535
max = 2126
over 1024 = 1070 examples (5.35%)
over 1280 = 150 examples (0.75%)
over 1536 = 20 examples (0.10%)
```

Conclusion: use `--max-seq-length 1536` for Qwen3.5 9B pilot training.

## Probe Results

One-step real-dataset probe:

```text
model = unsloth/Qwen3.5-9B
dataset = data/finetune/lichess_2000_2013-12_pilot.train.jsonl
max_seq_length = 1024
max_steps = 1
trainable parameters = 86,556,672 of 9,496,370,416
train_loss = 0.4474
output = outputs/finetune/qwen35_9b_lichess_2000_pilot_probe_lora
```

The 1024-token probe succeeded, but Unsloth removed 910/18,950 train examples
and 49/1,050 eval examples after truncation. This motivated the 1536-token
setting above.

Short speed probes at `max_seq_length=1536`:

```text
batch_size=1, grad_accum=1, max_steps=10:
train_runtime = 26.31s
train_samples_per_second = 0.38

batch_size=2, grad_accum=1, max_steps=10:
train_runtime = 48.41s
train_samples_per_second = 0.413
```

Expected 20k pilot wall-clock is roughly 12-13 hours on the current local stack.

## Full 20k Pilot Training

Started as a detached local process:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
.venv-train/bin/python -m finetune.train_lora \
  --model unsloth/Qwen3.5-9B \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot.train.jsonl \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_lora \
  --max-seq-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --eval-steps 0 \
  --save-steps 50 \
  --save-total-limit 2 \
  --logging-steps 5
```

Runtime state:

```text
pid file = outputs/finetune/qwen35_9b_lichess_2000_pilot_lora/train.pid
log file = outputs/finetune/qwen35_9b_lichess_2000_pilot_lora/train.log
python pid at start = 711400
num examples after filtering = 18,934
removed after 1536-token truncation = 16/18,950
total optimizer steps = 296
effective batch size = 64
observed speed = about 75-83 seconds per optimizer step
initial VRAM = about 12.2 GiB used / 11.9 GiB free
```

Final result:

```text
train_runtime = 23,450 seconds, about 6h 30m 50s
train_samples_per_second = 0.807
train_steps_per_second = 0.013
train_loss = 0.2282
epoch = 1.0
output = outputs/finetune/qwen35_9b_lichess_2000_pilot_lora
final adapter = outputs/finetune/qwen35_9b_lichess_2000_pilot_lora/adapter_model.safetensors
final checkpoint = outputs/finetune/qwen35_9b_lichess_2000_pilot_lora/checkpoint-296
```

The training process exited successfully and GPU memory returned to idle.

## Quick Held-Out Evaluation

Evaluated the trained LoRA and the base `unsloth/Qwen3.5-9B` on the same first
100 examples from:

```text
data/finetune/lichess_2000_2013-12_pilot.val.jsonl
```

Inference detail:

- `enable_thinking=False` is required for the Qwen3.5 chat template. This
  matches the training template, which emits an empty `<think></think>` block
  before the JSON answer.
- `max_new_tokens=24`
- greedy decoding, no sampling

Results:

```text
base Qwen3.5-9B:
json_parse_rate = 1.00
legal_move_rate = 0.51
top1_match_rate = 0.14

Qwen3.5-9B + pilot LoRA:
json_parse_rate = 1.00
legal_move_rate = 0.92
top1_match_rate = 0.22
```

Artifacts:

- `outputs/finetune/qwen35_9b_base_lichess_2000_pilot_val100_eval.json`
- `outputs/finetune/qwen35_9b_base_lichess_2000_pilot_val100_predictions.jsonl`
- `outputs/finetune/qwen35_9b_lora_lichess_2000_pilot_val100_eval.json`
- `outputs/finetune/qwen35_9b_lora_lichess_2000_pilot_val100_predictions.jsonl`

Monitor:

```bash
tail -f outputs/finetune/qwen35_9b_lichess_2000_pilot_lora/train.log
ps -p "$(cat outputs/finetune/qwen35_9b_lichess_2000_pilot_lora/train.pid)"
nvidia-smi
```

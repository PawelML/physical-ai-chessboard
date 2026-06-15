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

## Relabel with Stockfish (Distillation / Blunder Filtering)

`finetune/distill_dataset.py` relabels an existing JSONL with the arena's
`StockfishEvaluator` — prompts stay byte-identical (the target move is never in
the prompt), only the completion changes. Runs on CPU only and uses the normal
arena venv (`.venv`), not the training venv. Two modes:

- `--label-mode distill` (default): completion becomes Stockfish's best move at
  fixed `--nodes` (engine distillation, Variant B of the plan);
- `--label-mode filter`: keeps the human move but drops rows where it loses
  more than `--max-cpl` centipawns or misses a mate.

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
python -m finetune.distill_dataset \
  --input data/finetune/lichess_2000_2024-01_main.train.jsonl \
  --output data/finetune/lichess_2000_2024-01_distilled.train.jsonl \
  --metadata-output data/finetune/lichess_2000_2024-01_distilled.train.meta.json \
  --label-mode distill \
  --nodes 100000
```

Repeat for the `.val.jsonl` file so held-out metrics use the same labels.
Throughput is ~25 rows/s with the default worker count (≈2 h for the 190k main
set). Exact duplicate (prompt, completion) rows — common for openings once
labels are deterministic — are dropped by default (`--no-dedup` keeps them).
The sidecar metadata records human/engine agreement and human CPL stats, which
quantify the teacher noise (≈50% agreement for 2000-rated players). Use
`--limit 100` for a smoke run first.

Rationale and the decision record behind this step:
[`plans/fine_tune_model/pilot-analysis-and-distillation.md`](../plans/fine_tune_model/pilot-analysis-and-distillation.md).

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

## Phase 2 Gemma4 E4B Pilot Training

Gemma4 E4B uses a different chat template from Qwen. Keep the response masking
parts aligned with the tokenizer-rendered turns:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
python -m finetune.train_lora \
  --model unsloth/gemma-4-E4B-it-unsloth-bnb-4bit \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot.train.jsonl \
  --output-dir outputs/finetune/gemma4_e4b_lichess_2000_pilot_lora \
  --max-seq-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --eval-steps 0 \
  --save-steps 50 \
  --save-total-limit 2 \
  --logging-steps 5 \
  --instruction-part $'<|turn>user\n' \
  --response-part $'<|turn>model\n'
```

The 20k pilot run completed on the local RTX 3090 in 9287 seconds. Held-out
LoRA sanity on 100 examples produced 97% JSON parse, 37% legal moves, and 8%
top-1 match. This means the pipeline is valid, but the Gemma4 E4B pilot is much
weaker than the Qwen3.5 9B pilot on chess move quality.

## Phase 4 Export to GGUF and Ollama

Export the trained Qwen3.5 9B pilot adapter, write Ollama Modelfiles, and
optionally create the local Ollama models:

```bash
python -m finetune.export_ollama \
  --adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_lora \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf \
  --model-name chess-ft-qwen35-9b-pilot \
  --create-ollama
```

The default quantizations are `q4_k_m` and `q8_0`. If the GGUF files already
exist and only the Ollama import should be repeated, use:

```bash
python -m finetune.export_ollama \
  --skip-export \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf \
  --model-name chess-ft-qwen35-9b-pilot \
  --create-ollama
```

The generated Ollama models are named from the quantization suffix:

```text
chess-ft-qwen35-9b-pilot-q4_k_m
chess-ft-qwen35-9b-pilot-q8_0
```

Run a held-out sanity check through Ollama before arena benchmarking:

```bash
python -m finetune.evaluate_baseline \
  --dataset data/finetune/lichess_2000_2013-12_pilot.val.jsonl \
  --model chess-ft-qwen35-9b-pilot-q8_0 \
  --limit 100 \
  --timeout-seconds 300 \
  --num-ctx 4096 \
  --num-predict 24 \
  --think off \
  --progress-every 20
```

The Modelfile template intentionally wraps the raw arena prompt in the Qwen3.5
chat format used during training and emits the empty thinking block expected
when `enable_thinking=False`.

For Gemma4 E4B, pass the matching template family:

```bash
python -m finetune.export_ollama \
  --adapter-dir outputs/finetune/gemma4_e4b_lichess_2000_pilot_lora \
  --output-dir outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf \
  --model-name chess-ft-gemma4-e4b-pilot \
  --template-family gemma4 \
  --create-ollama
```

If Unsloth's GGUF conversion wrapper fails after merging, convert manually with
llama.cpp and then repeat the Ollama import with `--skip-export`:

```bash
python /home/pawelo/.unsloth/llama.cpp/convert_hf_to_gguf.py \
  --outfile outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-bf16.gguf \
  --outtype bf16 \
  outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf

/home/pawelo/.unsloth/llama.cpp/llama-quantize \
  outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-bf16.gguf \
  outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-q4_k_m.gguf \
  q4_k_m

/home/pawelo/.unsloth/llama.cpp/llama-quantize \
  outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-bf16.gguf \
  outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-q8_0.gguf \
  q8_0
```

The generated Gemma4 Ollama models are:

```text
chess-ft-gemma4-e4b-pilot-q4_k_m
chess-ft-gemma4-e4b-pilot-q8_0
```

## Phase 6 GRPO with Stockfish Reward

GRPO continues from the Stockfish-distilled Qwen3.5 policy, not from the base
model. The reward is computed with the arena's own `StockfishEvaluator` CPL and
the arena parser: malformed JSON is `-1.0`, illegal moves are `-0.5`, and legal
moves receive `1 - min(CPL, 300) / 300`.

First record the distilled SFT CPL baseline on the held-out validation split:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
python -m finetune.evaluate_cpl \
  --dataset data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl \
  --adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --output outputs/finetune/qwen35_9b_distilled_lora_val_cpl_eval.json \
  --predictions-output outputs/finetune/qwen35_9b_distilled_lora_val_cpl_predictions.jsonl \
  --limit 948 \
  --disable-thinking \
  --reward-nodes 50000
```

Run the first bounded GRPO smoke pass. If
`outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf` does not
exist, `train_grpo` creates it once from the distilled adapter with
`save_pretrained_merged(..., save_method="merged_16bit")`.

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
python -m finetune.train_grpo \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl \
  --distilled-adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --merged-model-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_smoke_lora \
  --limit 10000 \
  --max-steps 30 \
  --max-seq-length 1536 \
  --max-prompt-length 1536 \
  --max-completion-length 16 \
  --num-generations 8 \
  --temperature 1.0 \
  --learning-rate 1e-5 \
  --beta 0.04 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --reward-nodes 50000 \
  --logging-steps 5
```

The GRPO script writes:

- `grpo_config.json` with exact hyperparameters;
- `reward_metrics.jsonl` with mean reward, malformed fraction, illegal fraction,
  and legal-move CPL per reward callback;
- the final GRPO LoRA adapter and tokenizer in the output directory.

Only after the smoke run keeps JSON/legal stable, run the full one-epoch pass by
using a fresh output directory and `--max-steps -1`. Gate the result with the
same CPL eval command against the GRPO adapter; proceed to GGUF/Ollama and arena
games only if mean generated CPL drops without a JSON/legal regression.

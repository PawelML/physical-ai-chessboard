# GRPO Stockfish Reward Status

Date: 2026-06-13
Status: 1,000-step GRPO completed; tactical/error-mined/mixed pilots completed; no 3,000-step run yet

## Context Read

Read before implementation:

- `finetuning-chess-llm-plan.md`
- `phase0-status.md`
- `phase2-status.md`
- `pilot-analysis-and-distillation.md`
- `qwen35-distilled-pilot-status.md`
- `distilled-vs-human-100games-status.md`
- `grpo-stockfish-reward-plan.md`

Key conclusion carried forward: Stockfish distillation made the Qwen3.5 9B pilot
much more robust on format/legal moves, but did not improve the tactical metric
that matters for the arena, namely blunders/game and average CPL. GRPO is now
targeted at that CPL/blunder metric directly.

## Implemented Repo Changes

Added:

- `finetune/chess_reward.py`
  - Parses completions with `arena_core.parser.parse_uci_json`.
  - Checks move legality with `python-chess` from each row's stored `fen`.
  - Scores legal moves with
    `arena_core.evaluators.stockfish.StockfishEvaluator.evaluate_move`.
  - Uses reward ordering from the plan:
    - malformed JSON: `-1.0`
    - illegal move: `-0.5`
    - legal move: `1 - min(CPL, 300) / 300`
    - mate/no-CPL positions: `1.0` for Stockfish best move, else `0.0`
  - Includes `(fen, uci)` cache and optional multiprocessing pool with one
    persistent single-threaded Stockfish evaluator per worker.
  - Can be called directly as a TRL reward function and writes per-callback
    reward metrics when configured.
- `finetune/train_grpo.py`
  - Mirrors `finetune/train_lora.py` conventions: frozen config dataclass,
    argparse entrypoint, lazy imports, config sidecar, adapter/tokenizer save.
  - Starts from the distilled SFT policy by loading a merged distilled HF
    checkpoint and adding a fresh LoRA.
  - Auto-creates the merged checkpoint once from
    `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora` if missing.
  - Pre-renders GRPO prompts with the tokenizer chat template using
    `add_generation_prompt=True` and `enable_thinking=False`.
  - Keeps `fen` as a dataset column so TRL forwards it to the reward function.
  - Defaults to a bounded `--max-steps 30` smoke run before any full overnight
    run.
- `finetune/evaluate_cpl.py`
  - Generates greedy moves from a LoRA/HF checkpoint on a held-out JSONL.
  - Reports JSON parse rate, legal move rate, top-1 match, mean reward, and mean
    generated-move CPL.
  - Reuses the same Stockfish reward scorer so offline CPL and training reward
    use the same parser, legality check, and Stockfish CPL code path.
- `tests/test_finetune_chess_reward.py`
  - Unit coverage for malformed/illegal/legal reward ordering, CPL scaling,
    mate/no-CPL handling, and cache behavior.
- `finetune/README.md`
  - Adds Phase 6 commands for baseline CPL eval and bounded GRPO smoke run.

## Intended Smoke Command

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
.venv-train/bin/python -m finetune.train_grpo \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl \
  --distilled-adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --merged-model-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_smoke_lora \
  --limit 10000 \
  --max-steps 30 \
  --num-generations 8 \
  --temperature 1.0 \
  --learning-rate 1e-5 \
  --beta 0.04 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --reward-nodes 50000 \
  --logging-steps 5
```

## Intended Offline CPL Baseline

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
.venv-train/bin/python -m finetune.evaluate_cpl \
  --dataset data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl \
  --adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --output outputs/finetune/qwen35_9b_distilled_lora_val_cpl_eval.json \
  --predictions-output outputs/finetune/qwen35_9b_distilled_lora_val_cpl_predictions.jsonl \
  --limit 948 \
  --disable-thinking \
  --reward-nodes 50000
```

Repeat the same command with the GRPO adapter after the smoke/full run. The gate
is mean generated-move CPL down versus the distilled SFT baseline, with JSON
parse rate and legal-move rate not regressing.

## Local Environment Dependency Fix

Initial finding:

```text
trl 0.23.0
```

`GRPOConfig` imports, but `GRPOTrainer` currently fails during import with:

```text
No module named 'vllm'
```

This happens before any `use_vllm=False` runtime configuration can take effect.

Fix applied locally:

```bash
.venv-train/bin/python -m pip install "trl==0.24.0"
```

`pyproject.toml` now pins `trl==0.24.0` in the `train` extra.

`finetune/train_grpo.py` also normalizes tuple-valued TRL optional-dependency
flags after importing `trl` and before resolving `GRPOTrainer`, so the training
entrypoint does not rely solely on a hand-edited `.venv-train`.

One compatibility patch was also applied inside the local `.venv-train` TRL
install:

- file:
  `.venv-train/lib/python3.12/site-packages/trl/import_utils.py`
- reason: with `transformers==5.5.0`,
  `transformers.utils.import_utils._is_package_available()` returns
  `(available, version)` even when TRL calls it without `return_version=True`.
  TRL then treats `(False, None)` as truthy and imports missing optional
  packages such as `vllm` or `mergekit`.
- patch: wrap the Transformers helper so non-version calls return a real bool.

Post-fix import check:

```text
trl 0.24.0
is_vllm_available False bool
is_mergekit_available False bool
GRPOTrainer ok <class 'trl.trainer.grpo_trainer.GRPOTrainer'>
```

## Validation Performed

Code checks:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q \
  tests/test_finetune_chess_reward.py \
  tests/test_finetune_baseline_eval.py \
  tests/test_finetune_dataset_builder.py \
  tests/test_finetune_smoke_dataset.py
```

Result:

```text
All checks passed.
7 passed.
```

CLI import/help checks:

```bash
.venv-train/bin/python -m finetune.train_grpo --help
.venv-train/bin/python -m finetune.evaluate_cpl --help
```

Both entrypoints parse and print help successfully.

GRPO dependency dry-run:

```bash
rm -rf /tmp/grpo-dependency-check
ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish" \
.venv-train/bin/python -m finetune.train_grpo \
  --output-dir /tmp/grpo-dependency-check \
  --merged-model-dir /tmp/nonexistent-merged-checkpoint \
  --no-auto-merge-distilled \
  --max-steps 0 \
  --limit 1
```

Result: imports and configuration reached successfully; the script stopped only
at the expected pre-model-load guard:

```text
Merged distilled checkpoint is missing at /tmp/nonexistent-merged-checkpoint;
rerun without --no-auto-merge-distilled or create it manually.
```

Real Stockfish reward smoke:

```bash
.venv/bin/python - <<'PY'
from finetune.chess_reward import RewardConfig, StockfishRewardScorer

path = "vendor/stockfish/root/usr/games/stockfish"
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
with StockfishRewardScorer(RewardConfig(stockfish_path=path, reward_nodes=1000, workers=1)) as scorer:
    print(scorer.score_one(fen=fen, completion='{"move":"e2e4"}').to_json())
PY
```

Result:

```text
reward = 0.9933
parse_ok = True
legal_ok = True
parsed_move = e2e4
centipawn_loss = 2
best_move_uci = g1f3
classification = best
```

## GRPO Smoke Run

Date: 2026-06-13

Command:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
.venv-train/bin/python -m finetune.train_grpo \
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

Prep notes:

- The first attempt failed during merged checkpoint creation because `/` filled
  up and one safetensor shard was partial.
- Removed the incomplete merged checkpoint directory.
- Removed old `checkpoint-*` directories from prior LoRA runs, preserving final
  `adapter_model.safetensors` files.
- Removed generated Gemma4 pilot GGUF files to create enough free space for the
  Qwen merged checkpoint and GRPO smoke output:
  - `outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-q4_k_m.gguf`
  - `outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-q8_0.gguf`
- The merged distilled HF checkpoint was then created successfully:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf`.

Result:

- GRPO completed all 30 optimizer steps.
- Training runtime: 557.1 s.
- Train samples/s: 0.431.
- Train steps/s: 0.054.
- Final train loss: `1.625e-05`.
- Final adapter saved:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_smoke_lora/adapter_model.safetensors`
- Checkpoints saved:
  - `checkpoint-25`
  - `checkpoint-30`

Last logged trainer metrics at step 30:

```text
reward = 0.1954
reward_std = 0.3091
kl = 0.0006319
completion_length = 9
completions/clipped_ratio = 0
grad_norm = 0.8234
learning_rate = 2.931e-08
```

Reward metrics from `reward_metrics.jsonl`:

```text
batches = 30
examples = 240
malformed = 0
illegal = 55
legal = 185
malformed_fraction = 0.0
illegal_fraction = 0.2292
mean_reward = 0.2200
mean_legal_cpl_weighted = 284.83
```

Interpretation:

- Format stayed stable: no malformed JSON across 240 sampled completions.
- Completion length stayed fixed at 9 tokens; no clipped completions.
- KL stayed tiny, so the smoke did not show obvious policy drift.
- Illegal sampled moves were non-trivial at ~23% with `temperature=1.0`. This is
  not a hard smoke failure, but before the full run consider reducing exploration
  (`temperature` or `num_generations`) or keeping the run and relying on the
  `-0.5` illegal penalty while monitoring early full-run metrics closely.

Offline CPL gate:

```text
distilled SFT adapter, val limit 948:
json_parse_rate = 1.0000
legal_move_rate = 0.8544
top1_match_rate = 0.1530
mean_generated_cpl = 289.23
mean_reward = 0.3561
illegal = 138

GRPO smoke adapter, val limit 948:
json_parse_rate = 1.0000
legal_move_rate = 0.8608
top1_match_rate = 0.1477
mean_generated_cpl = 281.58
mean_reward = 0.3638
illegal = 132
```

Interpretation: the smoke adapter improved legal rate, mean reward, and mean
generated CPL while preserving strict JSON output. Top-1 Stockfish match dropped
slightly, so the next run should still be monitored rather than treated as a
final model.

Post-run code fix:

- The smoke process completed training and saved artifacts, but hung during
  `multiprocessing.Pool.join()` while shutting down reward workers.
- `finetune/chess_reward.py` now terminates the worker pool during scorer close
  to avoid that post-save hang on future runs.

## Bounded GRPO Pilot

Date: 2026-06-13

The first attempt at a full run used `--max-steps -1 --num-train-epochs 1`.
With the 10,000-example pilot dataset and TRL/Unsloth settings this expanded to
`Total steps = 10,000`, which would take roughly tens of hours at the observed
step time. That process was stopped after startup.

Next run uses an explicit 1,000-step cap:

```bash
setsid bash outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora/run_1000.sh \
  > outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora/train.log \
  2>&1 < /dev/null &
```

Output directory:
`outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora`.

Launch status:

- Started in the background as PID `1511796`.
- Log file:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora/train.log`
- PID file:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora/train.pid`
- Trainer startup confirmed `Total steps = 1,000`.
- Early aggregate after 11 reward batches / 88 sampled completions:
  - malformed: `0`
  - illegal: `19` (`0.2159`)
  - legal: `69`
  - mean reward: `0.2359`
  - weighted legal CPL: `312.99`

Completion status:

- Completed all 1,000 optimizer steps.
- Training runtime: `1.663e+04` seconds.
- Train samples/s: `0.481`.
- Train steps/s: `0.060`.
- Final train loss: `0.004852`.
- Final adapter saved:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora/adapter_model.safetensors`
- Retained checkpoints:
  - `checkpoint-900`
  - `checkpoint-1000`
- Final reward-metrics aggregate:
  - reward batches: `1000`
  - sampled completions: `8000`
  - malformed: `0`
  - illegal: `1007` (`0.1259`)
  - legal: `6993`
  - mean reward: `0.3776`
  - weighted legal CPL: `256.29`

Offline CPL gate after 1,000-step pilot:

```text
GRPO 1,000-step adapter, val limit 948:
json_parse_rate = 1.0000
legal_move_rate = 0.9262
top1_match_rate = 0.1371
mean_generated_cpl = 231.14
mean_reward = 0.4694
illegal = 70
cpl_scored_legal_moves = 833
```

Comparison against earlier gates:

```text
distilled SFT adapter:
legal_move_rate = 0.8544
mean_generated_cpl = 289.23
mean_reward = 0.3561
illegal = 138

GRPO smoke adapter:
legal_move_rate = 0.8608
mean_generated_cpl = 281.58
mean_reward = 0.3638
illegal = 132

GRPO 1,000-step adapter:
legal_move_rate = 0.9262
mean_generated_cpl = 231.14
mean_reward = 0.4694
illegal = 70
```

Interpretation: the 1,000-step pilot substantially improved legal rate,
mean reward, generated CPL, and illegal-move count on the held-out validation
slice. Top-1 Stockfish match dropped relative to SFT, so downstream arena
benchmarking is still required before treating this as a stronger chess player.

## Arena Benchmark

Date: 2026-06-13

Export:

- Merged GRPO LoRA into 16-bit HF with Unsloth `save_pretrained_merged`.
- Converted merged HF to BF16 GGUF with
  `$HOME/.unsloth/llama.cpp/convert_hf_to_gguf.py`.
- Quantized to Q8_0 with `$HOME/.unsloth/llama.cpp/llama-quantize`.
- Removed intermediate BF16 GGUF and merged HF after Q8_0 was created.
- Ollama model:
  `chess-ft-qwen35-9b-pilot-grpo-q8_0:latest`
- Final local GGUF:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_gguf/qwen35_9b_lichess_2000_pilot_grpo_1000.Q8_0.gguf`

Sanity check:

- `ollama run chess-ft-qwen35-9b-pilot-grpo-q8_0` on a real validation prompt
  returned strict JSON: `{"move":"g1f3"}`.

Benchmark setup:

- First attempted run `run_id=14` used `guidance_mode=legal_list`; it was
  cancelled after 2 completed games because run 12 used `strategic_memory`.
- Proper apples-to-apples run is `run_id=15`.
- Opponent/settings match run 12:
  - Stockfish 1320 beginner: `skill=2`, `limit_strength=true`, `target_elo=1320`
  - Stockfish eval: `nodes=200000`, `threads=1`, `hash_mb=128`
  - 100 games, 50 white + 50 black, seed 0
  - `legality_mode=constrained`
  - `guidance_mode=strategic_memory`
  - `ollama_thinking=true`
  - prompt `strict-v7`, `prompt_id=1`, `opening_suite_id=1`

Results:

| Metric | Distilled SFT (run 12) | GRPO 1000 Q8_0 (run 15) |
| --- | ---: | ---: |
| W-D-L | 0-4-96 | 0-3-97 |
| Lost by checkmate | 96 | 93 |
| Lost by `forfeit_invalid` | 0 | 4 |
| Draw claim | 4 | 3 |
| Blunders / game | 2.07 | 2.06 |
| Total blunders | 207 | 206 |
| Avg CPL | 141.44 | 106.10 |
| Accuracy rate | 0.4220 | 0.5163 |
| Illegal attempts | 14 / 2002 | 27 / 2386 |
| Illegal attempt rate | 0.0070 | 0.0113 |
| Malformed attempts | 0 | 0 |

Per-color:

| Metric | Run 12 black | Run 15 black | Run 12 white | Run 15 white |
| --- | ---: | ---: | ---: | ---: |
| W-D-L | 0-2-48 | 0-1-49 | 0-2-48 | 0-2-48 |
| Avg plies | 38.70 | 45.92 | 41.74 | 49.38 |
| Avg CPL | 142.65 | 111.67 | 140.33 | 100.97 |
| Blunders | 104 | 104 | 103 | 102 |
| Accuracy rate | 0.3996 | 0.4978 | 0.4426 | 0.5334 |
| Illegal attempts | 10 | 7 | 4 | 20 |
| Forfeits | 0 | 0 | 0 | 4 |

Interpretation:

- The offline CPL gate was directionally correct: GRPO reduced arena CPL by
  about 25% and improved Stockfish best/good move accuracy by about 9.4 points.
- The improvement did **not** convert to match strength against Stockfish 1320:
  wins stayed at 0, draws were roughly flat, and blunders/game were unchanged.
- GRPO appears to polish average move quality and lets games run longer, but it
  still makes about two decisive blunders per game.
- The 4 invalid forfeits are a regression versus run 12 and should be addressed
  before treating this as a deployable arena model. Likely cause: GRPO training
  optimized strict `{"move":"..."}` completions, while the arena strategic-memory
  benchmark asks for `move+rationale+strategy_update`.

## Legal-List Arena Benchmark

Date: 2026-06-13

Reason:

- The strategic-memory benchmark above is useful for arena realism, but it mixes
  move quality with a response-format mismatch: the GRPO policy was trained for
  strict move-only JSON, while strategic-memory asks for
  `move+rationale+strategy_update`.
- A clean policy comparison therefore needs the same move-only prompt contract
  used by GRPO training.

Setup:

- Opponent: Stockfish 1320 beginner.
- Games: 100 each, 50 white + 50 black.
- `legality_mode=constrained`
- `guidance_mode=legal_list`
- `temperature=0.0`
- `ollama_thinking=false`
- Prompt contract verified for run 17: strict `{"move":"e2e4"}` only; no
  `rationale`, no `strategy_update`, no strategic-memory text.

Runs:

- Distilled SFT Q8_0 baseline: `run_id=16`
- GRPO 1,000-step Q8_0: `run_id=17`

Results:

| Metric | Distilled SFT (run 16) | GRPO 1000 Q8_0 (run 17) |
| --- | ---: | ---: |
| W-D-L | 0-4-96 | 0-8-92 |
| Lost by checkmate | 93 | 91 |
| Lost by `forfeit_invalid` | 3 | 1 |
| Draw claim | 4 | 8 |
| Avg game plies | 38.30 | 47.26 |
| Avg CPL | 135.88 | 114.97 |
| Accuracy rate | 0.4101 | 0.4902 |
| Blunders / game | 2.02 | 2.03 |
| Total blunders | 202 | 203 |
| Mistakes | 229 | 239 |
| Inaccuracies | 282 | 343 |
| Illegal attempts | 26 / 1918 | 16 / 2358 |
| Illegal attempt rate | 0.0136 | 0.0068 |
| Malformed attempts | 0 | 0 |

Per-color:

| Metric | Run 16 black | Run 17 black | Run 16 white | Run 17 white |
| --- | ---: | ---: | ---: | ---: |
| W-D-L | 0-1-49 | 0-2-48 | 0-3-47 | 0-6-44 |
| Avg plies | 37.04 | 44.22 | 39.56 | 50.30 |
| Avg CPL | 142.76 | 113.58 | 129.53 | 116.17 |
| Accuracy rate | 0.4059 | 0.4738 | 0.4140 | 0.5044 |
| Blunders | 109 | 98 | 93 | 105 |
| Illegal attempts | 20 | 6 | 6 | 10 |
| Forfeits | 3 | 1 | 0 | 0 |

Interpretation:

- This clean move-only benchmark confirms the offline CPL signal: GRPO reduces
  average CPL by about 15.4%, improves accuracy by about 8.0 percentage points,
  halves illegal-attempt rate, doubles draws, and makes games materially longer.
- The same benchmark also confirms the main weakness: decisive tactical
  failures did not improve. Blunders/game is effectively flat at about 2.0.
- A longer run with the same reward is likely to further polish average move
  quality, but the available evidence does not prove it will reduce blunders.

Decision:

- Do not treat the current reward as ready for an immediate larger 3,000-step
  run if the target metric is lower blunders/game.
- Next training should first adjust reward shaping toward tactical safety:
  penalize high-CPL legal moves more sharply, add a large penalty for blunder
  threshold crossings, and consider emphasizing positions where the distilled
  policy or current GRPO policy makes decisive errors.
- A 3,000-step continuation is reasonable only after a short shaped-reward smoke
  shows blunders/game movement on a small arena or offline tactical gate.

## Tactical Reward Pilot

Date: 2026-06-14

Implemented changes:

- Added `reward_mode` to `finetune.chess_reward.RewardConfig`.
- Preserved the original linear reward as `reward_mode=linear`.
- Added `reward_mode=tactical`, which:
  - keeps CPL <= 20 at reward `1.0`;
  - gives good moves positive reward;
  - pushes mistake-range moves negative;
  - gives CPL > 300 legal blunders reward `-1.0`;
  - gives illegal moves reward `-1.0`, so illegal moves are not preferable to
    legal blunders.
- Added CLI support in `finetune.train_grpo` and `finetune.evaluate_cpl`.
- Extended CPL eval metrics with `blunders`, `mistakes`, `inaccuracies`, and
  their rates.
- Added focused unit coverage for the tactical reward ordering.

Validation:

```bash
.venv/bin/python -m ruff check \
  finetune/chess_reward.py \
  finetune/train_grpo.py \
  finetune/evaluate_cpl.py \
  tests/test_finetune_chess_reward.py
.venv/bin/python -m pytest -q tests/test_finetune_chess_reward.py
.venv-train/bin/python -m finetune.train_grpo --help
.venv-train/bin/python -m finetune.evaluate_cpl --help
```

Result:

```text
All checks passed.
4 passed.
```

Pilot command:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
.venv-train/bin/python -m finetune.train_grpo \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl \
  --distilled-adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --merged-model-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_tactical_300_lora \
  --limit 10000 \
  --max-steps 300 \
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
  --reward-mode tactical \
  --logging-steps 10 \
  --save-steps 100 \
  --save-total-limit 2
```

Output:

- Adapter:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_tactical_300_lora`
- Retained checkpoints:
  - `checkpoint-200`
  - `checkpoint-300`
- Runtime: about 5,001 seconds.
- Train samples/s: `0.48`.
- Train steps/s: `0.06`.
- Final train loss: `0.00141`.

Training reward aggregate:

| Window | Completions | Illegal | Illegal rate | Mean reward | Weighted legal CPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 300 batches | 2400 | 448 | 0.1867 | -0.0818 | 259.44 |
| Last 100 batches | 800 | 130 | 0.1625 | -0.0263 | 268.39 |
| Last 50 batches | 400 | 62 | 0.1550 | -0.0683 | 301.94 |

Offline CPL/blunder gate:

```bash
.venv-train/bin/python -m finetune.evaluate_cpl \
  --dataset data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl \
  --adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_tactical_300_lora \
  --output outputs/finetune/qwen35_9b_grpo_tactical_300_val948_cpl_eval.json \
  --predictions-output outputs/finetune/qwen35_9b_grpo_tactical_300_val948_cpl_predictions.jsonl \
  --limit 948 \
  --disable-thinking \
  --reward-nodes 50000 \
  --reward-mode tactical
```

Results against the same 948-example validation slice:

| Metric | Distilled SFT | GRPO 1000 linear | GRPO 300 tactical |
| --- | ---: | ---: | ---: |
| Legal move rate | 0.8544 | 0.9262 | 0.8966 |
| Illegal moves | 138 | 70 | 98 |
| Top-1 match rate | 0.1530 | 0.1371 | 0.1287 |
| Mean generated CPL | 289.23 | 231.14 | 236.40 |
| Blunder rate | 0.2785 | 0.2278 | 0.2278 |
| Blunders | 264 | 216 | 216 |
| Mistake rate | 0.0654 | 0.1065 | 0.0939 |
| Inaccuracy rate | 0.0876 | 0.1055 | 0.1023 |

Interpretation:

- The tactical reward works mechanically: illegal moves and legal blunders now
  both receive a hard `-1.0` signal.
- The 300-step tactical pilot improves substantially over distilled SFT on the
  offline gate.
- It does **not** beat the current GRPO 1000 linear adapter: blunder rate is tied,
  while legal rate and CPL are slightly worse.
- Because the previous offline blunder improvement did not transfer to arena
  blunders/game, this tactical-300 result is not strong enough to justify a
  3,000-step run or GGUF export by itself.

Decision:

- Do not export or arena-benchmark tactical-300 yet.
- Do not launch 3,000 steps yet.
- The next experiment should make the tactical signal more targeted before
  spending more GPU time: train on/error-mine positions where the policy blunders,
  or weight/curriculum the dataset toward high-risk tactical positions.

## Error-Mined Tactical Pilot

Date: 2026-06-14

Goal:

- Train on positions where the current arena models actually made high-CPL
  mistakes, instead of continuing on the broad generic training mix.
- Keep the experiment short and require it to beat GRPO 1000 linear offline
  before any export or arena benchmark.

Dataset construction:

```bash
.venv/bin/python -m finetune.build_error_mined_dataset \
  --db arena.db \
  --run-id 16 \
  --run-id 17 \
  --no-mate-missed \
  --output data/finetune/qwen35_9b_legal_list_runs16_17_high_cpl_error_mined_tactical.jsonl \
  --metadata-output data/finetune/qwen35_9b_legal_list_runs16_17_high_cpl_error_mined_tactical.meta.json
```

Notes:

- Source runs were the clean `legal_list` arena runs:
  - run 16: distilled Q8_0
  - run 17: GRPO 1000 Q8_0
- `mate_missed` was excluded from this first mined slice because the current
  mate classification can be ambiguous when `best_move_uci == accepted_uci`.
- The builder drops duplicate FENs and drops rows where Stockfish's best move
  equals the model's bad move.

Dataset stats:

```text
written = 667
candidates = 687
dropped_duplicate_fen = 18
dropped_best_equals_accepted = 2
by_classification = {"blunder": 509, "mistake": 158}
by_run = {"16": 339, "17": 328}
```

Training command:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
.venv-train/bin/python -m finetune.train_grpo \
  --train-dataset data/finetune/qwen35_9b_legal_list_runs16_17_high_cpl_error_mined_tactical.jsonl \
  --distilled-adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --merged-model-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf \
  --output-dir outputs/finetune/qwen35_9b_legal_list_error_mined_grpo_tactical_300_lora \
  --limit 667 \
  --max-steps 300 \
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
  --reward-mode tactical \
  --logging-steps 10 \
  --save-steps 100 \
  --save-total-limit 2
```

Output:

- Adapter:
  `outputs/finetune/qwen35_9b_legal_list_error_mined_grpo_tactical_300_lora`
- Retained checkpoints:
  - `checkpoint-200`
  - `checkpoint-300`
- Runtime: about 4,947 seconds.
- Train samples/s: `0.485`.
- Train steps/s: `0.061`.
- Final train loss: `0.002722`.

Training reward aggregate on the mined slice:

| Window | Completions | Illegal | Illegal rate | Mean reward | Weighted legal CPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 300 batches | 2400 | 36 | 0.0150 | -0.0983 | 255.36 |
| Last 100 batches | 800 | 6 | 0.0075 | 0.0285 | 217.83 |
| Last 50 batches | 400 | 1 | 0.0025 | 0.0404 | 207.86 |

Interpretation of training metrics:

- The model learned the mined slice mechanically: illegal sampling collapsed
  from about 1.5% overall to 0.25% in the last 50 batches, and mean reward
  became positive in the final windows.
- This alone is not enough; the gate is whether the adapter generalizes back to
  the held-out validation mix and eventually arena blunders/game.

Offline gate command:

```bash
.venv-train/bin/python -m finetune.evaluate_cpl \
  --dataset data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl \
  --adapter-dir outputs/finetune/qwen35_9b_legal_list_error_mined_grpo_tactical_300_lora \
  --output outputs/finetune/qwen35_9b_error_mined_grpo_tactical_300_val948_cpl_eval.json \
  --predictions-output outputs/finetune/qwen35_9b_error_mined_grpo_tactical_300_val948_cpl_predictions.jsonl \
  --limit 948 \
  --disable-thinking \
  --reward-nodes 50000 \
  --reward-mode tactical
```

Offline validation comparison:

| Metric | Distilled SFT | GRPO 1000 linear | GRPO 300 tactical | Error-mined tactical 300 |
| --- | ---: | ---: | ---: | ---: |
| Legal move rate | 0.8544 | 0.9262 | 0.8966 | 0.8418 |
| Illegal moves | 138 | 70 | 98 | 150 |
| Top-1 match rate | 0.1530 | 0.1371 | 0.1287 | 0.1635 |
| Mean generated CPL | 289.23 | 231.14 | 236.40 | 249.85 |
| Blunder rate | 0.2785 | 0.2278 | 0.2278 | 0.2373 |
| Blunders | 264 | 216 | 216 | 225 |
| Mistake rate | 0.0654 | 0.1065 | 0.0939 | 0.0854 |
| Inaccuracy rate | 0.0876 | 0.1055 | 0.1023 | 0.0717 |

Interpretation:

- The mined adapter overfits the high-risk slice but does not generalize better
  than GRPO 1000 linear.
- It improves top-1 match and reduces mistakes/inaccuracies, but it regresses
  legal rate, mean CPL, and blunder rate versus GRPO 1000 linear.
- This fails the gate for export, arena testing, or a 3,000-step run.

Decision:

- Do not export or arena-benchmark the error-mined tactical 300 adapter.
- Do not launch 3,000 steps from this setup.
- The next approach should mix mined tactical positions with the broad distilled
  dataset instead of training only on the mined slice, or introduce sampling
  weights/curriculum so the model keeps general legality and average move
  quality while seeing more high-risk positions.

## Mixed Broad + Error-Mined Tactical Pilot

Date: 2026-06-14

Reason:

- The pure error-mined run learned its own hard slice but lost too much general
  legality and average move quality.
- The next hypothesis was to mix normal broad examples with repeated high-CPL
  mined examples, so the model keeps its general policy while seeing tactical
  failures more often.

Dataset:

- Output:
  `data/finetune/qwen35_9b_mixed_broad3000_error_mined3x_tactical.jsonl`
- Metadata:
  `data/finetune/qwen35_9b_mixed_broad3000_error_mined3x_tactical.meta.json`
- Construction:
  - random seed: `0`
  - 3,000 sampled rows from
    `data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl`
  - 667 high-CPL mined rows repeated 3x = 2,001 rows
  - total rows: 5,001
  - approximate mix: 60% broad, 40% mined
- The JSONL schema was normalized before training because Hugging Face
  `datasets` rejected mixed object/string types in the original `source` field.

Training command:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
.venv-train/bin/python -m finetune.train_grpo \
  --train-dataset data/finetune/qwen35_9b_mixed_broad3000_error_mined3x_tactical.jsonl \
  --distilled-adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --merged-model-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf \
  --output-dir outputs/finetune/qwen35_9b_mixed_broad_error_mined_grpo_tactical_300_lora \
  --limit 5001 \
  --max-steps 300 \
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
  --reward-mode tactical \
  --logging-steps 10 \
  --save-steps 100 \
  --save-total-limit 2
```

Output:

- Adapter:
  `outputs/finetune/qwen35_9b_mixed_broad_error_mined_grpo_tactical_300_lora`
- Retained checkpoints:
  - `checkpoint-200`
  - `checkpoint-300`
- Runtime: about 4,978 seconds.
- Train samples/s: `0.482`.
- Train steps/s: `0.060`.
- Final train loss: `0.0006207`.

Training reward aggregate:

| Window | Completions | Illegal | Illegal rate | Mean reward | Weighted legal CPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 300 batches | 2400 | 276 | 0.1150 | -0.1095 | 275.07 |
| Last 100 batches | 800 | 74 | 0.0925 | 0.0065 | 270.72 |
| Last 50 batches | 400 | 34 | 0.0850 | -0.0225 | 283.87 |

Offline gate:

```bash
.venv-train/bin/python -m finetune.evaluate_cpl \
  --dataset data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl \
  --adapter-dir outputs/finetune/qwen35_9b_mixed_broad_error_mined_grpo_tactical_300_lora \
  --output outputs/finetune/qwen35_9b_mixed_broad_error_mined_grpo_tactical_300_val948_cpl_eval.json \
  --predictions-output outputs/finetune/qwen35_9b_mixed_broad_error_mined_grpo_tactical_300_val948_cpl_predictions.jsonl \
  --limit 948 \
  --disable-thinking \
  --reward-nodes 50000 \
  --reward-mode tactical
```

Offline validation comparison:

| Metric | Distilled SFT | GRPO 1000 linear | GRPO 300 tactical | Error-mined tactical 300 | Mixed tactical 300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legal move rate | 0.8544 | 0.9262 | 0.8966 | 0.8418 | 0.8766 |
| Illegal moves | 138 | 70 | 98 | 150 | 117 |
| Top-1 match rate | 0.1530 | 0.1371 | 0.1287 | 0.1635 | 0.1498 |
| Mean generated CPL | 289.23 | 231.14 | 236.40 | 249.85 | 259.53 |
| Blunder rate | 0.2785 | 0.2278 | 0.2278 | 0.2373 | 0.2521 |
| Blunders | 264 | 216 | 216 | 225 | 239 |
| Mistake rate | 0.0654 | 0.1065 | 0.0939 | 0.0854 | 0.0812 |
| Inaccuracy rate | 0.0876 | 0.1055 | 0.1023 | 0.0717 | 0.0833 |

Interpretation:

- Mixing broad and mined data improved legal rate versus pure error-mined
  tactical, but it still did not recover the legality/CPL of GRPO 1000 linear.
- It worsened blunder rate versus both GRPO 1000 linear and the broad tactical
  300 pilot.
- The current tactical/error-mined route is not yet a useful upgrade path.

Decision:

- Do not export or arena-benchmark the mixed tactical 300 adapter.
- Do not launch a 3,000-step tactical/error-mined run from these settings.
- The best current model remains GRPO 1000 linear Q8_0 for arena-facing
  experiments, despite its unchanged arena blunders/game.

## Next Execution Steps

1. Fix or audit mate-position classification before using `mate_missed` as
   training signal; current mined-data inspection showed ambiguous
   `best_move_uci == accepted_uci` cases.
2. Add an offline gate specifically on the mined high-CPL positions, separate
   from the broad validation gate, so we can measure specialization versus
   generalization explicitly.
3. If continuing GRPO, try continuation from the GRPO 1000 adapter rather than
   restarting every tactical pilot from distilled SFT.
4. Keep GRPO 1000 linear as the only candidate worth exporting/arena-testing
   until a new pilot beats it on broad offline blunder rate and legal rate.

## Mate Classification Fix and Tactical Gate

Date: 2026-06-14

Reason:

- The mined tactical path needed a cleaner error signal before using
  `mate_missed` examples.
- The previous Stockfish mate classification treated any smaller mate number
  as worse. That was wrong for winning mates: `mate_before=4` and
  `mate_after=3` means the move shortened the forced mate and should remain a
  mate position, not a missed mate.

Code changes:

- `arena_core/evaluators/stockfish.py`
  - Added `_mate_score_worsened`.
  - Winning forced mate is now missed only if the move loses the forced mate or
    flips into an incoming mate.
  - Incoming mate is worse only when the side to move gets mated sooner.
  - Walking from a non-mate position into an incoming mate is `mate_missed`.
- `tests/test_stockfish_evaluator.py`
  - Added unit coverage for shorter winning mate, losing forced mate, walking
    into mate, delaying incoming mate, and accelerating incoming mate.
- Existing arena DB mate labels from earlier runs should be treated as
  unreliable unless re-evaluated, because the classification bug affected both
  `mate_missed` and `mate_position` attribution.

DB audit after applying the corrected classifier to stored mate rows:

| Run | Mate rows | Old -> new: missed -> missed | missed -> position | position -> missed | position -> position |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 Distilled SFT | 817 | 147 | 188 | 231 | 251 |
| 17 GRPO 1000 Q8_0 | 836 | 168 | 183 | 265 | 220 |

Tactical gate:

- Builder:
  `finetune/build_tactical_gate.py`
- Output:
  `data/finetune/qwen35_9b_val948_tactical_gate_from_distilled_grpo1000.jsonl`
- Metadata:
  `data/finetune/qwen35_9b_val948_tactical_gate_from_distilled_grpo1000.meta.json`
- Source dataset:
  `data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl`
- Source predictions:
  - `outputs/finetune/qwen35_9b_distilled_lora_val948_cpl_predictions.jsonl`
  - `outputs/finetune/qwen35_9b_grpo_1000_val948_cpl_predictions.jsonl`
- Selection:
  - `--min-cpl 250`
  - include Stockfish `blunder` classifications
  - dedupe by FEN
- Stats:
  - dataset rows: 948
  - prediction rows scanned: 1,896
  - selected rows: 353
  - duplicate FEN rows dropped: 161
  - by classification: 332 blunders, 21 mistakes
  - by source: 275 distilled failures, 78 GRPO-1000 failures

Tactical gate evaluation command shape:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
.venv-train/bin/python -m finetune.evaluate_cpl \
  --dataset data/finetune/qwen35_9b_val948_tactical_gate_from_distilled_grpo1000.jsonl \
  --adapter-dir <adapter_dir> \
  --output outputs/finetune/<name>_tactical_gate353_cpl_eval.json \
  --predictions-output outputs/finetune/<name>_tactical_gate353_cpl_predictions.jsonl \
  --limit 353 \
  --disable-thinking \
  --reward-nodes 50000 \
  --reward-mode tactical
```

Tactical gate results:

| Metric | Distilled SFT | GRPO 1000 linear | GRPO 300 tactical | Error-mined tactical 300 | Mixed tactical 300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legal move rate | 0.9263 | 0.9603 | 0.9292 | 0.8895 | 0.9065 |
| Illegal moves | 26 | 14 | 25 | 39 | 33 |
| Top-1 match rate | 0.0510 | 0.0482 | 0.0680 | 0.0680 | 0.0680 |
| Mean generated CPL | 585.20 | 462.71 | 452.58 | 520.39 | 500.83 |
| Blunder rate | 0.7280 | 0.5949 | 0.5694 | 0.6034 | 0.5949 |
| Blunders | 257 | 210 | 201 | 213 | 210 |
| Mistake rate | 0.0850 | 0.1416 | 0.0963 | 0.0737 | 0.0935 |

Interpretation:

- GRPO 1000 linear is still the best balanced checkpoint: it has the highest
  legal rate on both broad validation and the tactical gate.
- GRPO 300 tactical specializes slightly on this gate: mean CPL improves by
  about 10.1 CPL versus GRPO 1000 linear and blunders drop from 210 to 201.
  That gain is too small to justify the broad validation regression, where the
  same adapter has worse legal rate and does not reduce broad blunder rate.
- Error-mined tactical and mixed tactical do not improve the decision. Both are
  worse than GRPO 1000 linear on legality and CPL; mixed ties gate blunder count
  but loses badly on legal rate.
- The tactical/error-mined pilots from distilled SFT are therefore not the next
  arena candidate.

Decision:

- Keep GRPO 1000 linear as the current best model for arena-facing work.
- Do not export or arena-benchmark the tactical/error-mined/mixed 300-step
  adapters.
- Do not launch a long 3,000-step tactical/error-mined run from distilled SFT.
- If continuing training, continue from the GRPO 1000 adapter and require a
  candidate to beat it on both gates:
  - broad validation: legal rate at least as high and broad blunder rate lower;
  - tactical gate: legal rate close to GRPO 1000 and lower mean CPL/blunder
    count.

## GRPO 1000 Continuation Mixed Tactical Pilot

Date: 2026-06-14

Reason:

- The distilled-start tactical pilots either hurt broad validation or failed to
  improve tactical failures enough.
- The next hypothesis was to continue from the stronger GRPO 1000 linear
  adapter, with a small tactical mix and lower learning rate.

Code changes:

- `finetune/train_grpo.py`
  - Added `--initial-adapter-dir` so GRPO can continue from an existing LoRA
    adapter instead of always creating a fresh LoRA on top of the distilled
    merged model.
  - Added a trainable-parameter guard; the GRPO-1000 continuation loaded
    `86,556,672` trainable parameters.
- `finetune/build_mixed_tactical_dataset.py`
  - Added a reproducible builder for broad-plus-tactical JSONL datasets.

Dataset:

- Output:
  `data/finetune/qwen35_9b_grpo1000_continuation_broad3000_tactical353x3.jsonl`
- Metadata:
  `data/finetune/qwen35_9b_grpo1000_continuation_broad3000_tactical353x3.meta.json`
- Construction:
  - random seed: `1`
  - 3,000 sampled rows from
    `data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl`
  - 353 tactical gate rows repeated 3x = 1,059 rows
  - total rows: 4,059
  - approximate mix: 74% broad, 26% tactical

Training command:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
.venv-train/bin/python -m finetune.train_grpo \
  --train-dataset data/finetune/qwen35_9b_grpo1000_continuation_broad3000_tactical353x3.jsonl \
  --initial-adapter-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_grpo_1000_lora \
  --merged-model-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_merged_hf \
  --output-dir outputs/finetune/qwen35_9b_grpo1000_continuation_mixed_tactical_150_lora \
  --limit 4059 \
  --max-steps 150 \
  --max-seq-length 1536 \
  --max-prompt-length 1536 \
  --max-completion-length 16 \
  --num-generations 8 \
  --temperature 1.0 \
  --learning-rate 5e-6 \
  --beta 0.04 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --reward-nodes 50000 \
  --reward-mode tactical \
  --logging-steps 10 \
  --save-steps 50 \
  --save-total-limit 2
```

Output:

- Adapter:
  `outputs/finetune/qwen35_9b_grpo1000_continuation_mixed_tactical_150_lora`
- Retained checkpoints:
  - `checkpoint-100`
  - `checkpoint-150`
- Runtime: `2,523` seconds.
- Train samples/s: `0.476`.
- Train steps/s: `0.059`.
- Final train loss: `0.004235`.

Broad validation comparison:

| Metric | GRPO 1000 linear | GRPO 1000 continuation mixed tactical 150 |
| --- | ---: | ---: |
| Legal move rate | 0.9262 | 0.9177 |
| Illegal moves | 70 | 78 |
| Top-1 match rate | 0.1371 | 0.1456 |
| Mean generated CPL | 231.14 | 235.13 |
| Blunder rate | 0.2278 | 0.2289 |
| Blunders | 216 | 217 |
| Mistake rate | 0.1065 | 0.1055 |
| Inaccuracy rate | 0.1055 | 0.1065 |

Tactical gate comparison:

| Metric | GRPO 1000 linear | GRPO 1000 continuation mixed tactical 150 |
| --- | ---: | ---: |
| Legal move rate | 0.9603 | 0.9490 |
| Illegal moves | 14 | 18 |
| Top-1 match rate | 0.0482 | 0.0510 |
| Mean generated CPL | 462.71 | 480.92 |
| Blunder rate | 0.5949 | 0.5892 |
| Blunders | 210 | 208 |
| Mistake rate | 0.1416 | 0.1445 |

Interpretation:

- The continuation did not preserve the main strength of GRPO 1000: legality
  dropped on both broad validation and tactical gate.
- Broad validation also regressed on CPL and blunders, so the candidate fails
  the primary gate.
- Tactical gate blunders improved slightly from 210 to 208, but CPL worsened
  materially and legal rate dropped from 0.9603 to 0.9490. That is not a useful
  tradeoff for arena play.
- Continuing from GRPO 1000 is technically supported now, but this specific
  74/26 broad/tactical mix with tactical reward and learning rate `5e-6` is not
  a better checkpoint.

Decision:

- Do not export or arena-benchmark
  `qwen35_9b_grpo1000_continuation_mixed_tactical_150_lora`.
- Keep GRPO 1000 linear as the current best model.
- If trying another continuation, make it more conservative:
  - lower tactical share, for example 10-15% instead of 26%;
  - lower learning rate, for example `2e-6`;
  - evaluate `checkpoint-100` before committing to longer runs, because the
    final 150-step adapter already failed broad validation.

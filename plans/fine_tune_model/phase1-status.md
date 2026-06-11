# Phase 1 Status - Build the Dataset

Status: completed
Date: 2026-06-11

## Summary

Phase 1 is complete. The repo now has a real dataset builder for Lichess PGN
sources and a baseline evaluator for the held-out split. Both the 20k pilot
dataset and the 200k main dataset have been generated from real Lichess dumps,
and base Ollama smoke-test model baselines have been recorded.

The invariant from the fine-tuning plan is preserved:

- prompts are rendered by `arena_core.prompts.build_strict_prompt`;
- prompt version remains `strict-v7`;
- completion targets remain exactly `{"move":"<uci>"}`;
- train/validation split is by game, not by sampled position.

## Implemented Repo Changes

Added dataset builder:

- `finetune/build_dataset.py`
  - Streams PGN through `chess.pgn.read_game`.
  - Filters to rated, blitz-or-slower games.
  - Requires both players to meet `--min-elo`, default 2000.
  - Requires normal termination and at least `--min-plies`, default 20.
  - Samples up to `--max-positions-per-game`, default 10, evenly across a game.
  - Writes separate train and validation JSONL files.
  - Records game/source metadata, FEN, SAN, UCI target, prompt version, prompt
    template hash, legality mode, split, game id, and ply.
  - Supports `--pgn -` for `zstdcat ... | python scripts/build_finetune_dataset.py`.
- `scripts/build_finetune_dataset.py`
  - Thin CLI wrapper around `finetune.build_dataset`.

Added held-out baseline evaluator:

- `finetune/evaluate_baseline.py`
  - Runs an Ollama model over validation JSONL examples.
  - Uses the arena's `parse_uci_json` parser.
  - Reports JSON-parse rate, legal-move rate, and top-1 match rate.
  - Can optionally write per-example predictions for inspection.

Updated docs:

- `finetune/README.md`
  - Adds Phase 1 dataset build command.
  - Adds base-model held-out evaluation command.

Added tests:

- `tests/test_finetune_dataset_builder.py`
  - Verifies filtering, position sampling, split labels, strict-v7 prompts, and
    UCI JSON completion targets.
- `tests/test_finetune_baseline_eval.py`
  - Verifies parse/legal/top-1 baseline metrics without requiring Ollama.

## Validation Performed

```bash
.venv/bin/python -m ruff check finetune/build_dataset.py finetune/evaluate_baseline.py scripts/build_finetune_dataset.py tests/test_finetune_dataset_builder.py tests/test_finetune_baseline_eval.py
.venv/bin/python -m pytest tests/test_finetune_dataset_builder.py tests/test_finetune_baseline_eval.py tests/test_finetune_smoke_dataset.py
.venv/bin/python scripts/build_finetune_dataset.py --help
.venv/bin/python -m finetune.evaluate_baseline --help
```

Result:

```text
All checks passed.
4 passed.
```

## Real Pilot Dataset

Source:

- `data/finetune/sources/lichess_db_standard_rated_2013-12.pgn.zst`
- Lichess standard rated PGN dump, December 2013.

Command:

```bash
zstdcat data/finetune/sources/lichess_db_standard_rated_2013-12.pgn.zst \
  | .venv/bin/python scripts/build_finetune_dataset.py \
      --pgn - \
      --train-output data/finetune/lichess_2000_2013-12_pilot.train.jsonl \
      --val-output data/finetune/lichess_2000_2013-12_pilot.val.jsonl \
      --metadata-output data/finetune/lichess_2000_2013-12_pilot.meta.json \
      --max-examples 20000 \
      --seed 0
```

Result:

```text
Wrote 18950 train and 1050 val examples from 2000/454151 kept games.
```

Artifacts:

- `data/finetune/lichess_2000_2013-12_pilot.train.jsonl` - 18,950 examples
- `data/finetune/lichess_2000_2013-12_pilot.val.jsonl` - 1,050 examples
- `data/finetune/lichess_2000_2013-12_pilot.meta.json`

Metadata summary:

```text
examples_total = 20000
games_kept = 2000
games_seen = 454151
skipped elo_below_threshold_or_missing = 327215
skipped time_control_too_fast_or_unknown = 124151
skipped non_normal_termination = 768
skipped too_few_plies = 17
```

Sanity-check summary:

```text
train rows = 18950
train games = 1895
train modes = open 13261, constrained 5689
val rows = 1050
val games = 105
val modes = open 733, constrained 317
prompt_versions = strict-v7 only
completion format checks = 20000/20000
train/val game_id overlap = 0
```

## Base Model Held-Out Baseline

Command:

```bash
.venv/bin/python -m finetune.evaluate_baseline \
  --dataset data/finetune/lichess_2000_2013-12_pilot.val.jsonl \
  --model qwen2.5:1.5b \
  --output outputs/finetune/qwen25_15b_base_lichess_2000_2013-12_pilot_eval.json \
  --predictions-output outputs/finetune/qwen25_15b_base_lichess_2000_2013-12_pilot_predictions.jsonl \
  --limit 1000 \
  --progress-every 100
```

Result:

```text
examples = 1000
json_parse_rate = 1.000
legal_move_rate = 0.320
top1_match_rate = 0.078
service_errors = 0
```

Artifacts:

- `outputs/finetune/qwen25_15b_base_lichess_2000_2013-12_pilot_eval.json`
- `outputs/finetune/qwen25_15b_base_lichess_2000_2013-12_pilot_predictions.jsonl`

## Main Dataset

Source:

- `https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst`
- Streamed directly through `curl | zstdcat`; the 30.1 GB compressed dump was
  not stored locally.

Command:

```bash
curl -L --fail --show-error https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst \
  | zstdcat \
  | .venv/bin/python scripts/build_finetune_dataset.py \
      --pgn - \
      --train-output data/finetune/lichess_2000_2024-01_main.train.jsonl \
      --val-output data/finetune/lichess_2000_2024-01_main.val.jsonl \
      --metadata-output data/finetune/lichess_2000_2024-01_main.meta.json \
      --max-examples 200000 \
      --seed 0
```

Result:

```text
Wrote 190470 train and 9530 val examples from 20000/345233 kept games.
```

Artifacts:

- `data/finetune/lichess_2000_2024-01_main.train.jsonl` - 190,470 examples
- `data/finetune/lichess_2000_2024-01_main.val.jsonl` - 9,530 examples
- `data/finetune/lichess_2000_2024-01_main.meta.json`

Metadata summary:

```text
examples_total = 200000
games_kept = 20000
games_seen = 345233
skipped elo_below_threshold_or_missing = 182494
skipped time_control_too_fast_or_unknown = 134884
skipped non_normal_termination = 6780
skipped unrated = 783
skipped too_few_plies = 292
```

Sanity-check summary:

```text
train rows = 190470
train games = 19047
train modes = open 133077, constrained 57393
val rows = 9530
val games = 953
val modes = open 6619, constrained 2911
prompt_versions = strict-v7 only
completion format checks = 200000/200000
train/val game_id overlap = 0
```

## Main Dataset Base Baseline

Command:

```bash
.venv/bin/python -m finetune.evaluate_baseline \
  --dataset data/finetune/lichess_2000_2024-01_main.val.jsonl \
  --model qwen2.5:1.5b \
  --output outputs/finetune/qwen25_15b_base_lichess_2000_2024-01_main_eval.json \
  --predictions-output outputs/finetune/qwen25_15b_base_lichess_2000_2024-01_main_predictions.jsonl \
  --limit 1000 \
  --progress-every 100
```

Result:

```text
examples = 1000
json_parse_rate = 1.000
legal_move_rate = 0.310
top1_match_rate = 0.061
service_errors = 0
```

Artifacts:

- `outputs/finetune/qwen25_15b_base_lichess_2000_2024-01_main_eval.json`
- `outputs/finetune/qwen25_15b_base_lichess_2000_2024-01_main_predictions.jsonl`

## Next Operational Step

Proceed to Phase 2 training. Use the 20k pilot dataset for the first real
training run, then use the 200k main dataset once the training recipe is sane.

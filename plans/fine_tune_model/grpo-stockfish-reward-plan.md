# GRPO with a Stockfish Reward — Execution Plan (for the implementing agent)

Date: 2026-06-13
Status: ready to implement
Executor: another agent (codex). This document is the spec — it must be followable
without the surrounding chat.

> Read first for context (do not re-derive their conclusions):
> - [pilot-analysis-and-distillation.md](pilot-analysis-and-distillation.md) — §9 is the
>   GRPO sketch this plan expands; §7 explains why imitation can't beat the teacher and
>   why that's fine here.
> - [distilled-vs-human-100games-status.md](distilled-vs-human-100games-status.md) — why
>   we stop scaling SFT and go to RL: distillation fixed legality (0 forfeits, <1% illegal)
>   but did **not** move blunders/CPL at 16.7k examples.

## 1. Goal

Take the distilled SFT model and improve **tactical move quality** (the blunder/CPL
metric SFT could not move) with reinforcement learning, using full-strength Stockfish as
the reward signal. This is a contextual-bandit-style RL: dense per-position reward, one
move per prompt, **no full-game credit assignment**.

Success is measured by the *unchanged* benchmark contract (strict-v7, UCI-only parser).
This is RL polish, not a miracle — set expectations at "measurably fewer blunders / lower
CPL, possibly the first real wins vs SF 1320", not "high win rate".

## 2. Non-negotiable invariants (same as the rest of the project)

- **Benchmark contract never moves.** strict-v7 prompt, UCI-only, `parse_uci_json`. Any
  gain must show up under the same prompts the arena sends. Do not bump `prompt_version`.
- **`python-chess` owns legality.** Reward computation builds the board from the stored
  `fen` and checks legality with `chess.Board.legal_moves` — never trust the model.
- **Reuse arena code for scoring.** Reward = a function of
  `arena_core.evaluators.stockfish.StockfishEvaluator.evaluate_move(...).centipawn_loss`,
  the exact CPL the leaderboard uses. Parse completions with
  `arena_core.parser.parse_uci_json`. Do **not** reimplement either.
- **`arena_core` stays web-framework-free and untouched** unless a genuine shared bug is
  found; all new code lives under `finetune/`.
- **Train-from rule is DIFFERENT here, on purpose.** SFT runs always go `base → dataset`
  for attribution (pilot-analysis §8). RL legitimately **continues from the distilled SFT
  policy** — that is the whole point (§9 step 1: RL needs a policy that already plays
  legally). Start GRPO from the distilled checkpoint, not from base.

## 3. Starting artifacts (already on disk)

- Distilled SFT LoRA adapter:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora/`
- Distilled pilot datasets (each row has `prompt`, `completion`, `move`, `fen`, `san`):
  - train: `data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl` (16,696 rows)
  - val:   `data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl` (948 rows)
- Stockfish binary: `vendor/stockfish/root/usr/games/stockfish`
  (`export ARENA_STOCKFISH_PATH=$PWD/vendor/stockfish/root/usr/games/stockfish`).
- Base model id used by SFT: `unsloth/Qwen3.5-9B`; chat-template parts
  `instruction_part="<|im_start|>user\n"`, `response_part="<|im_start|>assistant\n"`.

Environment (separate from the arena venv):
```bash
python -m venv .venv-train && source .venv-train/bin/activate
pip install -e ".[train]"     # datasets, transformers, unsloth; trl comes transitively
```
GRPO lives in `trl` (`GRPOTrainer`, `GRPOConfig`), already importable in this env.

## 4. The reward function — `finetune/chess_reward.py` (the novel, reusable piece)

Per sampled completion, given the position's `fen`:

1. `parsed = parse_uci_json(completion)`.
2. **Malformed** (`not parsed.parse_ok` or `parsed.move is None`) → reward `-1.0`.
3. Parse UCI; **illegal** (`chess.Move.from_uci` raises, or move ∉ `board.legal_moves`)
   → reward `-0.5`.
4. **Legal** → `cpl = StockfishEvaluator.evaluate_move(board, move).centipawn_loss`;
   reward `= 1.0 - min(cpl, CPL_CAP) / CPL_CAP` with `CPL_CAP = 300`.
   - Best move (cpl 0) → `+1.0`; a ≥300-CPL blunder → `0.0`.
   - If `cpl is None` (mate involved): `+1.0` if `move.uci() == eval.best_move_uci`,
     else `0.0`.

This ordering is deliberate: any legal move (`[0, 1]`) scores above any illegal (`-0.5`)
above any malformed (`-1.0`), so RL can never trade format/legality for a marginal CPL
gain — this is the anti-format-collapse pressure on the reward side (the KL anchor is the
other half). GRPO normalizes advantages within each group of `k` samples, so absolute
scale matters less than this ordering and the dense `[0,1]` gradient on move quality.

Performance (this is the throughput bottleneck — `k × N_positions × steps` evaluations):
- Reuse the `multiprocessing.Pool` + `_init_worker` pattern from
  `finetune/distill_dataset.py` (persistent single-threaded `StockfishEvaluator` per
  worker). The TRL reward callback receives a *list* of completions — score the batch
  across the pool.
- **Cache by `(fen, parsed_uci)`** (and cache the per-`fen` best move). Opening positions
  repeat heavily; caching cuts engine calls dramatically.
- Reward-time nodes can be lower than the distill run (try `--reward-nodes 50000` first;
  the distill used 100k). Measure rows/s and tune. `evaluate_move` runs *two* analyses
  per call, so this dominates wall-clock.

## 5. Training — `finetune/train_grpo.py`

Mirror `finetune/train_lora.py`'s shape: frozen `@dataclass` config, `argparse`, lazy
`import_module("unsloth"/"trl"/"datasets")` with the same "activate .venv-train" SystemExit
message, write `grpo_config.json` to the output dir, save adapter + tokenizer at the end.

Policy initialization (start from the distilled SFT, per §2):
- Preferred: load the **merged 16-bit distilled HF checkpoint** as the GRPO base and add a
  fresh LoRA on top. If a merged checkpoint isn't already on disk, produce it once with
  Unsloth `save_pretrained_merged` from the distilled adapter, then point GRPO at it.
- `FastModel.from_pretrained(merged_distilled, max_seq_length=1536, load_in_4bit=True)`
  then `FastModel.get_peft_model(...)` with the same LoRA targets as `train_lora.py`
  (`r=32, alpha=32, dropout=0`, all linear).

Dataset:
- Take a subset (~10k) of the distilled **train** prompts. Each GRPO row needs the model
  `prompt` and the `fen` (passed through to the reward fn as a dataset column — TRL forwards
  extra columns to reward functions as kwargs).
- **Pre-render the prompt with the chat template** exactly as eval/arena do: reuse the
  `_apply_chat_template(..., disable_thinking=True)` logic from `finetune/evaluate_lora.py`
  (`add_generation_prompt=True`, `enable_thinking=False`). Store the rendered string as the
  GRPO `prompt` so the served prompt is byte-identical to inference. Do not let GRPO
  re-template a raw string.

`GRPOConfig` starting hyperparameters (tune from here):
- `num_generations = 8` (k samples per prompt — the GRPO group).
- `temperature = 1.0` (SFT policy is peaked; need exploration diversity).
- `max_completion_length = 16` (completion is ~10 tokens `{"move":"e2e4"}`).
- `max_prompt_length = 1536` (verify the p99 of rendered prompts; constrained prompts carry
  the legal list).
- `learning_rate = 1e-5` (RL needs a far lower LR than SFT's 2e-4; too high → format collapse).
- `beta = 0.04` (KL coefficient to the reference/SFT model — the format anchor; **do not set
  to 0**).
- `per_device_train_batch_size` small (1–2) + `gradient_accumulation_steps` to taste; 9B
  4-bit + sampling 8 completions is memory-heavy on 24 GB. Use gradient checkpointing.
- `optim="adamw_8bit"`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.03`, `seed=0`,
  `report_to="none"`, `logging_steps=5`.
- Generation backend: start with the default/Unsloth path (`use_vllm=False`); vLLM speeds
  rollouts but likely won't co-fit with 4-bit training on a single 3090 — only revisit if
  throughput is unacceptable.
- Scale: ~10k prompts is an overnight job on the 3090, bottlenecked by reward scoring, not
  the gradient step. Use `--max-steps` to bound a first short smoke run (e.g. 30 steps) and
  confirm the loop + reward + checkpointing work before the full run.

Log per step (not just reward): mean reward, **fraction malformed**, **fraction illegal**,
mean legal-move CPL. A rising malformed/illegal fraction is the early warning for format
collapse → stop, raise `beta`, lower `lr`.

## 6. Offline evaluation (fast feedback, before any arena games)

The existing `finetune/evaluate_lora.py` reports JSON/legal/top-1 but **not CPL** — and CPL
is the metric that matters. Extend it (add a `--score-cpl` flag) or add
`finetune/evaluate_cpl.py` that, on the held-out val set, generates the greedy move per
position and reports **mean CPL of generated moves** + legal-move rate + JSON rate, reusing
`StockfishEvaluator`.

Gate before spending arena hours:
- **mean generated-move CPL drops** vs the distilled SFT baseline on the same val set, AND
- JSON parse rate ≈ 100% and legal rate ≥ the SFT baseline (no format/legality regression).

Compute the distilled SFT baseline CPL once on the same val set for the comparison (the
adapter at `qwen35_9b_lichess_2000_pilot_distilled_lora`).

## 7. Export + arena benchmark (the payoff)

1. Merge the GRPO adapter and export to GGUF/Ollama. **Use the documented workaround** (the
   `finetune.export_ollama` Unsloth path failed last time): convert the merged HF model with
   llama.cpp `convert_hf_to_gguf.py` to `Q8_0`, then `llama-quantize --allow-requantize` for
   `Q4_K_M`. See the "Ollama Export" section of
   [qwen35-distilled-pilot-status.md](qwen35-distilled-pilot-status.md). Register e.g.
   `chess-ft-qwen35-9b-pilot-grpo-q8_0:latest`.
2. ⚠️ The Ollama `TEMPLATE` must match training (Qwen chat template, thinking disabled) —
   reuse the same Modelfile/template as the distilled export; a mismatch silently degrades
   output. Sanity-check: `ollama run <model>` on one real rendered prompt → bare `{"move":...}`.
3. Benchmark vs the **same opponent and config as run 12** so it's apples-to-apples.
   ⚠️ **run 12 was NOT launched from the CLI.** The `arena tournament` command has no
   game-count option and names runs differently; run 12 came from the **backend live
   Stockfish-match endpoint** with `game_count=100`. Reproduce it exactly via the backend:

   ```bash
   # Start the backend pointed at the same DB, with the same Stockfish engine settings.
   # Stockfish nodes/hash come from ARENA_* settings, NOT the request body — run 12 used
   # nodes=200000, hash_mb=128 (the config defaults). Keep them identical.
   export ARENA_DATABASE_URL="sqlite+aiosqlite:///$PWD/arena.db"
   export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
   uvicorn backend.main:app --port 8000 &

   # POST the same payload run 12 used (only `model` changes):
   curl -sS -X POST http://localhost:8000/matches/stockfish/start \
     -H 'content-type: application/json' \
     -d '{
       "model": "chess-ft-qwen35-9b-pilot-grpo-q8_0:latest",
       "stockfish_level": "beginner",
       "game_count": 100,
       "legality_mode": "constrained",
       "temperature": 0.0,
       "guidance_mode": "legal_list"
     }'
   # then rebuild aggregates and read the new run:
   arena rebuild-summaries
   ```

   Run 12's exact fingerprint to match (from `arena.db`):
   - opponent: `stockfish_level="beginner"` ⇒ skill 2, `UCI_LimitStrength`, target_elo 1320;
     engine `nodes=200000`, `threads=1`, `hash_mb=128` (from `ARENA_STOCKFISH_*` settings).
   - `game_count=100` (50 white + 50 black, color_policy "both"), `legality_mode=constrained`,
     `guidance_mode=legal_list`, seed 0, prompt strict-v7 (`prompt_id=1`),
     `opening_suite_id=1`.
   - sampler: `temperature=0.0`, `top_p=null`, `num_ctx=32768`, `num_predict=512`,
     `format=json`, `think=auto`, `cpu_offload_gpu_layers=48` — i.e. backend defaults +
     default `ARENA_OLLAMA_*`. Keep these env vars at their defaults so the model_snapshot
     fingerprint differs from run 12 only by the model weights.

   Headline comparison: blunders/game and avg CPL vs `run_id=12`, plus win rate ± Wilson CI.

## 8. Deliverables / definition of done

- `finetune/chess_reward.py` — reward fn + worker pool + `(fen, uci)` cache (reuses
  `StockfishEvaluator`, `parse_uci_json`).
- `finetune/train_grpo.py` — GRPO trainer mirroring `train_lora.py` conventions.
- CPL offline eval (extend `evaluate_lora.py` or new `evaluate_cpl.py`).
- `finetune/README.md` — a "Phase 6: GRPO with Stockfish reward" section with the exact
  commands run (mirror the existing phase sections).
- `plans/fine_tune_model/grpo-status.md` — status log written *as it runs* (env, dataset
  subset, hyperparameters, training curve incl. malformed/illegal fractions, offline CPL
  before/after, arena 100-game table vs run 12), in the same style as the other phase-status
  files.
- Checks pass: `ruff check .` (new files included; 100-char lines, rules E,F,I,UP,B,ASYNC).
  `mypy` scope is `arena_core backend tests` only, so `finetune/` is out of strict mypy, but
  keep type hints explicit and consistent with `train_lora.py`. Run `pytest -q` if any shared
  code is touched.
- Do **not** commit local DBs, GGUF/merged weights, or machine-specific paths. Commits are
  short imperative summaries scoped to one behavior.

## 9. Risks & abort conditions

| Symptom | Action |
| --- | --- |
| Malformed/illegal fraction climbs during training (format collapse) | Stop. Raise `beta` (KL), lower `lr`, confirm malformed=-1.0/illegal=-0.5 penalties are applied. |
| Reward throughput too low (rollouts starve the GPU) | Lower `--reward-nodes`, ensure the `(fen,uci)` cache is hit, reduce prompt subset, increase pool workers. |
| Mean reward rises but offline CPL doesn't drop (reward hacking) | Inspect generated moves; verify CPL is computed on the parsed legal move, not a cached wrong key. |
| Offline CPL drops but arena win rate flat | Acceptable outcome — report it honestly (blunders/game vs run 12 is the headline; "polish, not a miracle" was the stated expectation). |
| OOM on the 3090 | Smaller `per_device_train_batch_size`, gradient checkpointing on, `max_prompt_length` trimmed to measured p99, `num_generations` 8→6. |

## 10. Time budget (3090, order-of-magnitude)

| Step | Wall-clock | Notes |
| --- | --- | --- |
| Merge distilled adapter → 16-bit HF (if not on disk) | ~10–20 min | once |
| GRPO **smoke** run (`--max-steps 30`) | ~30–60 min | validate loop/reward/checkpointing first |
| **GRPO full run** (~10k prompts, k=8) | **~8–14 h** | the dominant, mostly-unattended item; bottleneck is rollout generation + Stockfish reward |
| Offline CPL eval (baseline + GRPO on 948 val) | ~30 min | gate before arena |
| Export merge → GGUF Q8+Q4 → Ollama | ~30–45 min | budget +1–2 h buffer: the Unsloth GGUF path failed last time (workaround in the distilled status doc) |
| Arena 100 games vs SF 1320 | ~1–1.5 h | run 12 was 1.1 h at ~40 s/game |

**Total compute ≈ 12–18 h**, dominated by one overnight GRPO run; most is unattended.
Implementation/debugging (writing `chess_reward.py` + `train_grpo.py`, tuning) is on top of
this and interactive. If reward throughput is the bottleneck, lower `--reward-nodes` and lean
on the `(fen, uci)` cache before reaching for vLLM.

## 11. One-paragraph summary for the executor

Continue from the distilled SFT Qwen3.5-9B policy (it already plays legally, which is why
this is now possible). Add a fresh LoRA and run GRPO on ~10k strict-v7 prompts: sample 8
moves per position, reward each with `1 - min(CPL,300)/300` for legal moves (`-0.5` illegal,
`-1.0` malformed) using the arena's own `StockfishEvaluator`, with a KL anchor (`beta=0.04`)
and low LR (`1e-5`) to keep the JSON format intact. Gate on held-out generated-move CPL
dropping below the distilled SFT baseline with no JSON/legality regression, then export to
Ollama and benchmark 100 games vs Stockfish 1320 in `constrained` mode, comparing
blunders/game and CPL against arena `run_id=12`.

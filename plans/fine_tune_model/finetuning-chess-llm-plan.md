# Fine-Tuning a Local Model to Beat Stockfish 1320 — Step-by-Step Plan

> Expansion of Option C in [beating-stockfish-analysis.md](beating-stockfish-analysis.md).
> Written for someone doing their first LLM fine-tune. Hardware assumption:
> one RTX 3090 (24 GB VRAM) locally, with optional cheap cloud GPU rental.
>
> **Goal:** a LoRA fine-tune of a local model that, served through Ollama and
> benchmarked by the arena *with the unchanged strict-v7 prompt contract*,
> (a) virtually never forfeits on illegal moves and (b) wins a measurable share
> of games against Stockfish beginner (Skill 2, `UCI_LimitStrength` 1320).
>
> **Why it's achievable:** DeepMind trained a 270M-param transformer to ~2300
> Elo with no search ("Grandmaster-Level Chess Without Search", 2024);
> gpt-3.5-turbo-instruct plays ~1750 Elo on PGN completion. An 8–12B model has
> orders of magnitude more capacity than needed for 1320-level chess — the
> current failure is *format + lack of chess-specific training*, not capacity.

## 0. The big picture

```
Lichess PGN dump ──► filter games ──► replay with python-chess ──► render the
(CC0, free)          (Elo, time      sample positions             EXACT strict-v7
                      control)                                    prompt per position
                                                                       │
                                                                       ▼
arena benchmark ◄── ollama create ◄── GGUF convert ◄── merge LoRA ◄── QLoRA train
(base vs FT,        (Modelfile +      + quantize        into base     (Unsloth,
 same harness)       chat template)   (Q8_0/Q4_K_M)                    3090 or cloud)
```

The single most important design decision: **training examples are rendered by
`arena_core.prompts.build_strict_prompt` itself**, not by a hand-written
imitation. The model trains on byte-identical prompts to the ones the arena
sends at benchmark time, and the completion target is exactly
`{"move":"<uci>"}`. No prompt-version bump, no contract change — the benchmark
measures the same thing before and after.

## 1. Phase 0 — Environment + smoke test (1 evening)

Goal: prove the whole pipeline end-to-end with toy sizes before investing in
data or training time.

1. Create a separate venv/conda env for training (don't pollute the arena env):
   `pip install unsloth` (pulls torch, transformers, peft, trl, bitsandbytes).
   Unsloth is the recommended tool — single-GPU, beginner-friendly, fastest
   QLoRA on consumer cards. Alternatives: LLaMA-Factory, axolotl.
2. Pick a *tiny* model for the smoke test (any ~1–4B instruct model available
   both on HF and in Ollama).
3. Hand-build ~2k training examples from one small PGN file (Phase 2 script in
   miniature), train 30 minutes, merge, convert to GGUF, load into Ollama, play
   one arena game against `random`. The point is to hit every tooling problem
   (CUDA versions, GGUF conversion, chat template) while iterations are cheap.

## 2. Phase 1 — Build the dataset (2–3 evenings)

### 2.1 Source data

- **Lichess open database** — https://database.lichess.org/ — monthly PGN
  dumps, CC0 licensed. One month is tens of GB compressed and far more games
  than needed; stream-decompress with `zstdcat`, never fully extract.
- Shortcut worth considering: the **Lichess Elite Database** (pre-filtered
  2400+ games) skips most of the filtering work.

### 2.2 Filtering (script: e.g. `scripts/build_finetune_dataset.py`)

Stream games with `chess.pgn.read_game` and keep a game when:

- rated; time control blitz or slower (bullet games are noisy);
- **both** players ≥ 2000 Elo (this is the "teacher strength" knob — see
  ablations in §6);
- normal termination (no abandonment), ≥ 20 plies.

### 2.3 Position sampling → training examples

For each kept game, replay it move by move and sample up to ~10 positions
(uniformly across the game so openings/middlegames/endgames are all covered).
For each sampled position, render the prompt with the real arena code:

```python
from arena_core.prompts import build_strict_prompt
built = build_strict_prompt(
    board=board,                    # position BEFORE the move
    san_history=san_so_far,
    own_moves=mover_prior_moves,    # (san, uci) pairs of the side to move
    last_opponent_move=last_opp_san,
    legality_mode=mode,             # see mix below
)
example = {"prompt": built.rendered, "completion": f'{{"move":"{move.uci()}"}}'}
```

Details that matter:

- **Target is `move.uci()` from python-chess** — this automatically gets
  castling (`e1g1`) and promotion (`e7e8q`) encodings right. Never format UCI
  strings by hand.
- **Legality-mode mix:** render ~70% of examples in `open` mode and ~30% in
  `constrained` mode, so the fine-tune is benchmarkable in both tracks. (Open
  prompts are also shorter ⇒ cheaper training tokens.)
- **Split train/val by game, not by position** — positions from one game are
  heavily correlated; splitting by position leaks.
- Store as JSONL. Record the generator config (Elo cutoff, month, sampling
  params) in a sidecar file — this is benchmark provenance, same ethos as
  `config_hash`.

### 2.4 Sizes

| Dataset | Examples | Purpose |
|---|---|---|
| pilot | 20k | first real training run, hyperparameter sanity |
| main | 200k | the headline run |
| scale-up (optional) | 500k–1M | only if the 200k curve hasn't flattened |

### 2.5 Pre-training baseline (do not skip)

Before training anything, run the base model over ~1k held-out examples and
record: legal-move rate, top-1 match rate vs the human move, JSON-parse rate.
These three numbers are the "before" picture and your progress meter — much
faster feedback than full arena games.

### Variant B (upgrade, optional): Stockfish-distilled labels

Instead of imitating human moves, label each sampled position with Stockfish's
best move at fixed nodes (the arena's evaluator wrapper already does this).
Cleaner signal, no human blunders, and labelling is CPU-only (an overnight
local job — no GPU needed). DeepMind's result used exactly this kind of
distillation. Recommended as a second iteration after Variant A works, and it
makes a great ablation chapter ("human imitation vs engine distillation").

## 3. Phase 2 — Train on the RTX 3090 (days, mostly unattended)

### 3.1 Model choice

- **Cleanest portfolio story:** fine-tune the HF checkpoint of a model already
  on your leaderboard (the gemma 12B-instruct class you benchmark) — then the
  base-vs-FT comparison is apples-to-apples on the existing leaderboard.
  Check Unsloth supports the architecture before committing.
- **Fastest iteration:** a 7–8B instruct sibling. Recommended for the first
  real run; do the 12B once the recipe works.
- Always the **instruct** variant (it already follows the JSON contract
  sometimes; you're sharpening, not teaching from scratch).

### 3.2 What fits in 24 GB

| Run | Fits on 3090? | Notes |
|---|---|---|
| 7–8B QLoRA (4-bit base) | Yes, comfortably (~12–16 GB) | seq 1024, batch 4–8 |
| 12B QLoRA | Yes, tight (~20–23 GB) | seq 1024, batch 1–2 + grad accum, gradient checkpointing |
| 27B QLoRA | No | cloud (≥48 GB) |
| Full fine-tune ≥7B | No | cloud (80 GB+), and unnecessary — LoRA suffices here |

### 3.3 Starting hyperparameters (QLoRA via Unsloth)

- LoRA: `r=32`, `alpha=32`, dropout 0, target all linear projections.
- LR `2e-4`, cosine schedule, warmup 3%; 1–2 epochs over the dataset.
- `max_seq_length=1024` (strict-v7 open-mode prompts with ASCII board are
  ~400–700 tokens; constrained adds the legal list — verify the 99th percentile
  on your real JSONL and trim the budget to it).
- Effective batch ≈ 64 via gradient accumulation.
- **Loss on completion tokens only** (Unsloth's `train_on_responses_only`).
  The completion is ~10 tokens, so per-example signal is small — that's why the
  dataset is 200k examples, and it's normal.
- Use the model's **own chat template** when formatting (Unsloth's
  `get_chat_template`); prompt = user turn, completion = assistant turn.

### 3.4 Wall-clock expectations

Rough planning numbers for a 3090 (QLoRA, seq 1024): pilot 20k examples on an
8B ≈ 1–2 h; main 200k on an 8B ≈ overnight-to-a-day per epoch; 12B roughly 2×
that. Treat these as order-of-magnitude — measure on the pilot and extrapolate.

### 3.5 During training

- Eval every N steps on the held-out set with a small callback computing the
  three Phase-1 metrics (legality / top-1 / JSON rate), not just loss.
- Expected trajectory: JSON rate → ~100% almost immediately; legality climbs
  fast; top-1 match plateaus somewhere around 40–55% vs 2000-rated humans —
  that's fine, you don't need to imitate perfectly to beat 1320.

## 4. Phase 3 — When to rent a cloud GPU (optional, cheap)

You can complete the whole plan without this. Rent when you want: the 12B
trained comfortably / faster, a 27B attempt, several ablations in parallel, or
Variant B at scale.

- **Providers:** RunPod, Vast.ai, Lambda — per-minute billing, no commitment.
- **Ballpark prices (verify current):** 48 GB (L40S/A6000) ≈ $0.7–1.2/h;
  A100 80 GB ≈ $1.5–2/h; H100 ≈ $2–3.5/h.
- **Cost reality:** a 200k-example QLoRA run is a few GPU-hours ⇒ **$5–30 per
  run**; the entire project including ablations should stay under ~$100.
- **Workflow:** build the JSONL locally (data work is CPU), upload dataset +
  the same training script, train, download only the LoRA adapter (tens of MB).
  Keep nothing precious on the rented box; assume it can vanish.

## 5. Phase 4 — Merge → GGUF → Ollama (1 evening)

1. Merge the LoRA into the base weights (Unsloth `save_pretrained_merged`).
2. Convert with llama.cpp's `convert_hf_to_gguf.py`, then quantize **both
   Q8_0 and Q4_K_M** — quantization is already a `model_snapshot` dimension in
   the arena, so benchmarking both is a free extra result.
3. `ollama create chess-ft -f Modelfile` pointing at the GGUF.
4. ⚠️ **Gotcha #1 of the whole project:** the Ollama `TEMPLATE` must reproduce
   the exact chat template used in training. A mismatch (missing system turn,
   wrong special tokens) silently degrades the model back to garbage output.
   Copy the template from the base model's existing Ollama manifest
   (`ollama show --modelfile <base>`) rather than writing one from scratch.
5. Sanity check before any benchmark: `ollama run chess-ft`, paste one real
   rendered prompt from your JSONL, expect bare `{"move":"..."}` back.

## 6. Phase 5 — Benchmark in the arena (the payoff)

- Same `prompt_version` (strict-v7), same Stockfish pin, same opening suite.
- Baseline: base model vs Stockfish beginner, **≥ 50 games per color per
  legality mode** (Wilson intervals are wide below that; `low_sample` flags it).
- Fine-tune: identical config. The leaderboard's snapshot fingerprinting makes
  the comparison rigorous by construction.
- Headline metrics: win rate (± Wilson CI), `forfeit_invalid` count, illegal
  rate, avg CPL.
- **Ablations worth running** (each is one tournament + one chart):
  - data Elo cutoff 1600 vs 2000 vs 2400 (does a stronger teacher help?);
  - dataset size 20k vs 200k (scaling curve);
  - Q8_0 vs Q4_K_M of the same fine-tune;
  - open vs constrained mode (does the fine-tune close the gap between them?);
  - Variant A (human imitation) vs Variant B (Stockfish distillation).
- If the calibrated-opponent ladder (Option A) exists by then, report the
  fine-tune's **anchored Elo estimate**, not just "beats/loses to 1320".

### Expected outcome (set honest expectations)

Behaviour cloning from 2000-rated humans typically yields play meaningfully
below the teacher but far above the current ~700–900 level — comfortably
enough to score real wins against a 1320 opponent. If the first run only
draws/rarely wins, the levers are (in order): more data, stronger teacher,
Variant B distillation. Do **not** reach for changing the benchmark contract.

## 7. Risks and gotchas

| Risk | Mitigation |
|---|---|
| Chat-template mismatch between training and Ollama serving | §5 step 4; sanity-check with a real prompt before benchmarking |
| Hand-formatted UCI breaks castling/promotion | always `move.uci()` from python-chess |
| Format drift (model stops emitting bare JSON) | loss on completion only; completion is always exactly `{"move":"..."}` |
| Val-set leakage | split by game, never by position |
| Overfitting to openings | sample positions across all game phases |
| Cloud box disappears mid-run | checkpoint adapters periodically; dataset lives locally |
| Licensing | Lichess data is CC0 ✓; check the base model's license permits fine-tuned redistribution if you publish weights |
| Overclaiming in the write-up | report Wilson CIs and sample sizes, anchor Elo via the ladder, state quantization |

## 8. Portfolio deliverables

1. `docs/finetuning-experiment.md` — method, before/after table, ablation
   charts, honest limitations section.
2. The dataset-builder script + training script in the repo (reproducibility
   is the brand of this project).
3. Leaderboard screenshot: base and fine-tuned snapshots side by side, same
   harness, Wilson CIs visible.
4. The interview pitch, one sentence: *"My benchmark showed local models lose
   to a 1320-rated engine mostly by checkmate at ~150 CPL; I built a LoRA
   pipeline on Lichess data rendered through the benchmark's own prompt
   contract, and the same harness now measures the fine-tune beating that
   engine — here's the scaling curve."*

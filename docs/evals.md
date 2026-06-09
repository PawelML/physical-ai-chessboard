# Evaluation & Benchmark Design — Software Arena (Phase 1)

> The valuable product is not that LLMs can play chess; it is that every move,
> failure, retry, cost, latency, prompt version and engine evaluation is
> **reproducible and queryable**. Treat this as an experiment runner.

## Metric categories (do not collapse into one score)

### Rule-following (primary signal in *open* mode)
- malformed-response rate
- illegal-move rate
- retries per accepted move
- retry success rate
- forfeit-by-invalid rate
- JSON-compliance rate

### Chess quality (judged by Stockfish)
- centipawn loss (CPL) vs Stockfish best
- blunder / mistake / inaccuracy counts
- win / draw / loss
- mate missed / mate allowed
- average evaluation swing after the model's move

### Operational
- latency per attempt and per accepted move
- prompt / completion tokens; context used / remaining
- cost per game (nullable; ~0 for local v1)
- timeout rate; provider error rate
- **model-swap latency** (when a VRAM-oversized Ollama pair forces reloads)

## Leaderboard philosophy

There is **no single "best model" score**. The leaderboard is a filterable table
with separate views for **open** vs **constrained** (combining them is meaningless).
Provide at least: legality score, chess-quality score, cost-normalized score,
latency-normalized score. Rows are keyed by **immutable model snapshot**.

Required filter dimensions (also in [`data-model.md`](data-model.md)):
model snapshot · run · strict/reasoning · open/constrained · opponent format ·
opponent identity · color · opening suite/version · Stockfish version+options ·
sampler config · prompt version · quantization · context window.

## Tournament design

- **model-vs-model** — natural gameplay.
- **model-vs-Stockfish (skill-capped)** — comparable, tunable strength.
- Both colors for every opening line; **version the opening suite**; report
  per-opening *and* aggregate. Opening choice can dominate results — avoid an
  all-tactical or all-mainline suite.

### How many games (signal)
- Portfolio / directional signal: **20–50 games per model per mode**.
- Credible comparison: **100+ games per pairing/opening suite**.
- Rule-following rates accumulate fast — every move attempt is a sample.

## Stockfish as judge (configuration)

- Pin the binary/version. **Single thread** for reproducibility.
- **Fixed nodes** (more reproducible across machines than fixed time).
- Store actual nodes used and depth reached, plus all UCI options.
- Evaluate the position **before and after** each accepted move → CPL.

## Determinism traps (capture or be wrong)

- `temperature=0` is **not** full determinism for local LLMs. Capture seed (if
  supported), Ollama version, model digest, quantization, sampler params, context size.
- **Mate scores**: never naively convert mate → centipawns. Store mate separately;
  define explicit classification rules for mate positions.
- **Skill-capped Stockfish** is not deterministic unless tightly configured —
  capture skill / `UCI_LimitStrength` / target Elo / nodes / seed.
- Opening suites dominate outcomes — both colors, versioned, balanced, reported per line.
- Token counts are **approximate** across heterogeneous local models — never exact.

## Example report (target shape)

```md
# Eval Report — run 2026-06-xx

Mode: open · Opponent: model-vs-model · Suite: starter-v1 (both colors)
Stockfish: 17 (1 thread, 200k nodes) · Prompt: strict-v6 · temperature=0

| Model snapshot        | Games | Illegal % | Avg retries | Avg CPL | Blunders | W-D-L  |
| --------------------- | ----: | --------: | ----------: | ------: | -------: | -----: |
| qwen3.5:9b @<digest>  |    30 |      4.2% |        0.11 |     142 |       18 | 11-6-13|
| gemma3n:e4b @<digest> |    30 |     12.7% |        0.34 |     210 |       29 | 13-6-11|
```

## Tie-in to later phases

The same metrics framework extends to the robot phase (occupancy verification
accuracy, calibration error, recovery rate, VLM agreement) — see
[`plans/embodied-ai-chess-lab-plan.md`](../plans/embodied-ai-chess-lab-plan.md).

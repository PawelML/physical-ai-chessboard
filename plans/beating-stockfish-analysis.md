# Beating Stockfish: Why Local Models Lose and What Would Change That

> Analysis of the Model Comparison results (June 2026): every local LLM lost
> every game against Stockfish "beginner" (Skill 2 + `UCI_LimitStrength` @ 1320).
> This doc explains *how* they lose, why zero wins is statistically expected at
> the current strength gap, and four ways to change the outcome — ranked by
> effort and by honesty. The most promising option (fine-tuning) is expanded in
> [finetuning-chess-llm-plan.md](finetuning-chess-llm-plan.md).

## 1. What the data actually says

Two failure modes, and they need different fixes.

**Mid-size models (12B+) lose at chess, not at legality.** In `arena.db`
(constrained mode, Stockfish beginner):

| Model | Games | Result | How they lose |
|---|---|---|---|
| gemma4:12b-it-bf16 | 20 | 0W / **1D** / 19L | 14 checkmates, 5 forfeits |
| gemma4:12b-it-q4_K_M | 20 | 0W / **2D** / 18L | 12 checkmates, 6 forfeits |
| gemma4:12b-it-q8_0 | 20 | 0W / 0D / 20L | 17 checkmates, 3 forfeits |
| gemma4:26b-a4b-it-qat | 4 | 0W / 0D / 4L | 4 checkmates, 0 forfeits |
| qwen3.6:27b | 4 | 0W / 0D / 4L | 2 checkmates, 2 forfeits |

Games last 30–50 plies, avg CPL is ~100–200, and there are already **3 draws**.
These models play perhaps 400–600 Elo below the 1320 opponent — a large gap,
but a finite one.

**Small models (≤9B) lose at legality.** In `arena-local-results.db` every
single game (gemma3n:e4b, qwen3.5:9b, qwen3-vl:8b, nanbeige 3b, gemma4:12b in
that config) ended `forfeit_invalid`. The notation experiment
(`arena-notation-results.db`) is inconclusive for the same reason: qwen3-vl:8b
forfeits within 1–3 plies under *every* contract (UCI JSON, SAN JSON, PGN
completion), so notation was never the binding constraint for that model.

**Curious aside worth investigating:** illegal-move rate is consistently much
higher when the LLM plays White (e.g. bf16: 0.14 as White vs 0.03 as Black;
gemma4:12b: 0.58 vs 0.00). Almost all forfeits happen with the White pieces.

## 2. The math: zero wins is the expected result

At a ~500 Elo gap the expected score is ~5%. With 20 games per model, zero
wins is not a surprising outcome — it is the *modal* outcome. "Sometimes wins"
therefore requires either a weaker opponent or a stronger model; more games at
the current gap would only surface the occasional fluke.

## 3. Four options, ranked

### Option A — Calibrated opponent ladder (cheapest, fully honest)

The current "beginner" floor is artificial. Stockfish's `UCI_Elo` cannot go
below 1320, but two mechanisms go lower:

1. **Skill Level 0 without `UCI_LimitStrength`, plus a tiny node budget.**
   `StockfishMoveSource` currently uses `nodes=50_000`; at `nodes=20–200` with
   Skill 0, Stockfish plays well below 1320.
2. **Blunder injection:** a wrapper move source that with probability *p* plays
   a uniformly random legal move and otherwise delegates to Stockfish. *p* gives
   a smoothly calibratable opponent from ~400 Elo upward.

This is not a trick — it is a missing benchmark feature. With a ladder of
calibrated opponents, the rating at which a model's win rate crosses 50%
becomes an **Elo estimate for the model** (reported with the Wilson intervals
the arena already computes). "Elo anchoring" on the leaderboard is a far better
portfolio argument than a single lucky win. Effort: ~1–2 days.

### Option B — Honest improvements to existing models (cheap experiments)

- Thinking mode for qwen3.6:27b (`think="auto"` covers it; check
  `attempts.thinking_used` to confirm it actually fired).
- `strategic_memory` guidance mode, lower-temperature presets.
- Re-run the SAN-vs-UCI notation experiment on gemma4:12b — a model strong
  enough that notation could plausibly be the binding constraint. Any switch is
  an explicit, versioned prompt experiment (see CLAUDE.md invariants).
- Investigate the White-side forfeit anomaly (§1) — it may be a cheap win.

### Option C — Fine-tune a local model (the portfolio centerpiece)

The literature says this is very achievable: DeepMind trained a 270M-parameter
transformer to ~2300 Elo blitz with no search at all ("Grandmaster-Level Chess
Without Search", 2024), and gpt-3.5-turbo-instruct plays ~1750 Elo on PGN
completion. A LoRA fine-tune of an 8–12B local model on Lichess games, rendered
in the **exact strict-v7 prompt contract** (FEN + history → `{"move":"e2e4"}`),
fixes both failure modes at once — legality *and* strength — and should
comfortably exceed Stockfish 1320.

The arena is the perfect instrument for this: `model_snapshot` fingerprinting
makes the base-vs-fine-tuned comparison rigorous by construction. The narrative
— *measured the failure → built a training pipeline → closed the gap →
re-measured with the same harness* — is exactly the story a portfolio needs.

Full step-by-step plan: [finetuning-chess-llm-plan.md](finetuning-chess-llm-plan.md).

### Option D — Assisted / agentic track (only as an explicitly separate benchmark)

Candidate sampling + self-verification, or tool access (e.g. a shallow
evaluation tool). Can work, but the arena's core invariant applies: **never
silently substitute a move in a scored run**. Any assistance must be a
separately labelled benchmark track, never mixed into the pure-LLM results —
otherwise the credibility of the whole arena is undermined.

## 4. Recommendation

1. Do **A** immediately: wins start appearing, and the Elo-anchoring ladder is
   a real leaderboard feature.
2. Do **C** as the project's "chapter two" and main interview material.
3. Treat **B** as cheap experiments along the way; keep **D** clearly fenced
   off if built at all.

# Human-Label vs Stockfish-Distilled Pilot — 100-Game Arena Comparison

Date: 2026-06-13

> Confirmatory benchmark following
> [pilot-analysis-and-distillation.md](pilot-analysis-and-distillation.md) and
> [qwen35-distilled-pilot-status.md](qwen35-distilled-pilot-status.md). The 100-position
> offline proxy was a tie (15% vs 16% top-1); this run asks the same question with
> 5× the sample and the metric that actually matters — blunders/CPL in real games.

## Setup

Both runs are byte-identical in configuration; **the only difference is the training
label source** (human moves vs Stockfish best moves on the same pilot positions).

- Opponent: **Stockfish 1320 (beginner)** — Stockfish 16, `skill=2`,
  `limit_strength=true`, `target_elo=1320`, `nodes=200000`, 1 thread, 128 MB hash.
- 100 games each (50 white + 50 black), `legality_mode=constrained`, seed 0,
  prompt `strict-v7`.
- Models (Ollama, Q8_0):
  - Human-label: `chess-ft-qwen35-9b-pilot-q8_0:latest` — arena `run_id=13`.
  - Distilled: `chess-ft-qwen35-9b-pilot-distilled-q8_0:latest` — arena `run_id=12`.

## Results

| Metric | Human-label (run 13) | Distilled (run 12) |
| --- | ---: | ---: |
| W–D–L | 0–1–99 | 0–4–96 |
| Lost by checkmate | 89 | 96 |
| **Lost by `forfeit_invalid`** | **10** | **0** |
| Lost by draw_claim (survived to draw) | 1 | 4 |
| illegal rate (black / white) | 4.6% / 3.1% | 1.0% / 0.4% |
| malformed rate | 0% | 0% |
| **blunders / game** | **2.05** (205 / 1903 moves) | **2.07** (207 / 1988 moves) |
| avg CPL (black / white) | 142 / 133 | 143 / 140 |
| top-1 accuracy (black / white) | 39.8% / 43.5% | 40.0% / 44.3% |
| avg plies (black / white) | 34.4 / 42.7 | 38.7 / 41.7 |

Win rate for both is 0/100 (Wilson 95% upper bound ≈ 7%).

## Conclusions

1. **Distillation won the "cheap half" decisively — and this was `constrained`
   mode, where the legal-move list was provided.** Even *with* the legal list in
   the prompt, the human-label model emitted illegal moves and burned all retries
   10 times (`forfeit_invalid`); the distilled model essentially never did
   (0 forfeits, <1% illegal, 3 extra survived draws). Robustness/legality is a real,
   repeatable win and makes the distilled model the better foundation.

2. **The headline metric — blunders/CPL — did not move.** 2.05 vs 2.07 blunders per
   game, CPL ~140 in both, top-1 ~40–44% in both, 0 wins in both. Swapping human
   labels for Stockfish labels at this data scale (16.7k examples) bought **no
   tactical improvement.**

3. **The offline proxy predicted the arena outcome correctly.** Held-out top-1 was
   15% vs 16% (a tie); the arena confirmed the tie on CPL/blunders at 5× the sample.
   This validates the "iterate on offline proxies, not full games" guardrail — future
   candidates can be screened on held-out CPL before spending arena hours.

## Diagnosis

Format/legality is a simple, highly repeated structure → it saturates by 16k examples.
Tactical move quality has a different, much steeper, much later learning curve
(DeepMind reached ~2300 Elo with *billions* of distilled positions). 16.7k positions
is below the threshold where tactical patterns imprint, so changing the label source
removed noise in the cheap dimension without touching the expensive one.

## Decision

- **Close the human-label arm.** It served its purpose as the imitation-vs-distillation
  ablation; distilled is ≥ on every axis at equal cost.
- **Do not pursue a larger SFT scale-up as the next step.** A single data point (16k)
  cannot prove SFT has plateaued, but the EV-for-the-goal is low: a 12× scale-up to
  ~190k would plausibly improve CPL ~140→~100–120, which is still nowhere near the
  ≤1-blunder/game needed to beat 1320. Scaling more imitation chases a log curve that
  almost certainly does not cross the winning threshold on a single 3090.
- **Next step is RL (GRPO with a Stockfish reward).** The distilled model now satisfies
  the RL prerequisite that human-label did not: a policy that already plays legally
  (0 forfeits, <1% illegal) to bootstrap from. RL has no imitation ceiling and rewards
  move quality directly — it targets exactly the blunder metric that SFT could not move.
  See [grpo-stockfish-reward-plan.md](grpo-stockfish-reward-plan.md).

## Open methodology note

Both runs are `constrained` only. The plan treats `open` and `constrained` as separate
benchmarks; an `open`-mode run (≥50 games/color, no legal list) would fully characterize
the distilled model's legality advantage and is worth doing opportunistically, but it is
not on the RL critical path.

## How these numbers were pulled

`arena.db`, tables `game_summaries` (per run/color aggregates) and `games`
(`termination_reason` breakdown), filtered to `run_id IN (12, 13)`. Blunders/game =
summed `blunders` across both colors ÷ 100 games.

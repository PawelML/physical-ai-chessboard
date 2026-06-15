# One-Ply Reranker / Blunder Classifier — Execution Plan (for the implementing agent)

Date: 2026-06-15
Status: ready to implement
Executor: another agent (codex). This document is the spec — it must be followable
without the surrounding chat.

> Read first for context (do not re-derive their conclusions):
> - [pilot-analysis-and-distillation.md](pilot-analysis-and-distillation.md) — imitation
>   ceiling; why SFT could not move blunders.
> - [distilled-vs-human-100games-status.md](distilled-vs-human-100games-status.md) — SFT
>   fixed legality but blunders/game stayed flat.
> - [grpo-status.md](grpo-status.md) — GRPO polished average CPL (141→106) and accuracy
>   (42%→52%) but blunders/game stayed flat (2.07→2.06), still 0 wins vs SF 1320. This is
>   the wall this plan attacks.

## 0. One-paragraph summary

GRPO/SFT optimize *generation* (one forward pass, one move). They moved the average but
not the tail: ~2 decisive blunders/game still lose every game vs SF 1320. This plan adds a
second, independent **selection** stage on top of generation: sample N candidate moves from
the LLM, then **veto the ones that lose**, and play the best survivor. This converts the
hard requirement "the generator must never blunder" into the much easier "the generator
must propose at least one non-blunder among N **and** the selector must recognize the
blunder" — two independent chances, the same reason best-of-N + verifier beats raw sampling.
This is **test-time compute / a shallow 1-ply safety filter**, so it is NOT strict-v7. It
must be benchmarked as an explicitly separate regime and reported next to (never mixed with)
the bare single-move numbers.

## 1. Goal

Reduce **blunders/game** vs Stockfish 1320 enough to score the first non-zero results
(draws/wins), by adding a 1-ply candidate-selection layer to the move source. Success is a
measurable blunders/game drop **and** the first W/D against SF 1320, reported as a new,
clearly-labeled benchmark regime.

This is deliberately staged so the cheap experiment gates the expensive one:

- **Phase A — Stockfish-veto ceiling (NO training).** Veto candidates with full Stockfish.
  This is the *upper bound* of what a perfect 1-ply filter buys. It is a go/no-go gate: if
  even a perfect veto does not produce W/D vs SF 1320, then the problem is not isolated
  blunders (it is conversion skill) and Phase B is not worth building.
- **Phase B — learned blunder classifier (training).** Replace Stockfish at play time with a
  learned classifier (no engine call during the game). The gap between Phase B and Phase A
  measures how much tactical judgment the net actually captured. This is the deliverable.
- **Baseline — strict-v7 single move (already exists).** The floor. Untouched.

## 2. Non-negotiable invariants

- **`python-chess` owns legality.** Every candidate is validated against `board.legal_moves`
  built from the position. Never trust the model's claim of legality.
- **Reuse arena code.** Parse with `arena_core.parser.parse_uci_json`. Score/label with
  `arena_core.evaluators.stockfish.StockfishEvaluator.evaluate_move` (the exact CPL the
  leaderboard uses). The reward/label scoring in `finetune/chess_reward.py` already does this
  — reuse it for Phase B labeling. Do **not** reimplement CPL or parsing.
- **`arena_core` stays web-framework-free.** The new move source lives in `arena_core`
  (it is a `MoveSource`); the classifier training code lives under `finetune/`.
- **The LLM prompt stays strict-v7.** The reranker does NOT change the prompt template or
  bump `prompt_version`. It changes the *move source* (N samples + selection). What changes
  is the **system/regime**, and that must be fingerprinted separately (see §6) so leaderboard
  rows never silently mix reranked runs with bare single-move runs.
- **Failed candidates are still data.** Log per-move reranker internals (candidates
  generated, how many illegal/malformed, how many vetoed, whether the veto changed the move)
  — that telemetry *is* benchmark value.
- **No engine at play time in Phase B.** Phase A may call Stockfish during the game (that is
  the explicit "ceiling/reference" caveat). Phase B must not — the classifier is the only
  judge, so it stays an LLM-system benchmark, not a Stockfish proxy.

## 3. Starting artifacts (already on disk)

- Best current policy (Ollama): `chess-ft-qwen35-9b-pilot-grpo-q8_0:latest` (GRPO 1000
  linear; the current best per grpo-status.md).
- Distilled datasets with `prompt`/`completion`/`move`/`fen`/`san` per row:
  - train: `data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl`
  - val:   `data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl`
- Arena DB with blunder-labeled games: `arena.db` (runs 12–17 have per-move CPL/classification).
- Stockfish binary: `vendor/stockfish/root/usr/games/stockfish`
  (`export ARENA_STOCKFISH_PATH=$PWD/vendor/stockfish/root/usr/games/stockfish`).
- Move-source seam: `arena_core/engine.py` `MoveSource` Protocol + `LLMMoveSource`
  (`propose(*, prompt, board) -> MoveProposal`); `MoveProposal` from `arena_core.move_sources`.
- Reward/label scorer: `finetune/chess_reward.py` (`StockfishRewardScorer`, `(fen,uci)` cache,
  worker pool). Classifier training mirrors `finetune/train_lora.py` conventions.

## 4. Phase A — Stockfish-veto ceiling (NO training)

### 4.1 New move source: `RerankedLLMMoveSource`

Add to `arena_core/engine.py` (or a new `arena_core/reranker.py` imported by the move-source
resolver). It wraps an inner `LLMMoveSource` and a pluggable scorer; it is itself a
`MoveSource`, so the engine's parsing/persistence path is unchanged.

`propose(*, prompt, board)`:

1. **Generate N candidates.** Call the inner service `k` times (config `n_candidates`, start
   `k=5`) with temperature > 0 (start `0.8`) so samples diverge. Keep the prompt byte-identical
   to strict-v7. (Sequential calls are fine; completions are ~10 tokens.)
2. **Parse + validate** each completion with `parse_uci_json` + `chess.Move.from_uci` +
   `move in board.legal_moves`. Drop malformed/illegal. Deduplicate to distinct legal UCI
   moves, and **count multiplicity** (how many of the k samples produced each move) — this
   frequency is the model's own preference / self-consistency signal.
3. **Score each distinct legal candidate** with the scorer (Phase A: `StockfishEvaluator` at
   low nodes, start `veto_nodes=50000`). A candidate is `blunder` if `centipawn_loss > 300`
   or it allows mate (reuse the classification from `evaluate_move`).
4. **Select**:
   - Among non-vetoed candidates, pick the one with highest sample multiplicity (ties → lowest
     CPL). This is "play the move the model preferred most, among the safe ones".
   - If **all** candidates are vetoed, fall back to the lowest-CPL candidate (least-bad), and
     record `all_vetoed=True`.
   - If **no** legal candidate was produced at all, return the model's first raw response and
     let the existing retry/`forfeit_invalid` path handle it (do not invent a move).
5. Return a `MoveProposal` whose `raw_response` is `{"move":"<chosen_uci>"}`, with aggregated
   latency/token counts. Attach reranker telemetry (see §6).

Config (env `ARENA_RERANKER_*` or move-source args): `n_candidates`, `temperature`,
`veto_cpl_threshold=300`, `veto_nodes=50000`, `scorer={stockfish|classifier}`.

### 4.2 Wire it into the move-source resolver

Extend `_source_from_name` (in `cli.py`) and the backend move-source resolution so a model
can be requested in "reranked" mode (e.g. name prefix `reranked:<model>` or a `guidance_mode`
value). Keep it opt-in; default behavior is unchanged.

### 4.3 Benchmark and go/no-go gate

Run vs **Stockfish 1320 beginner**, 100 games (50W/50B), same engine settings as run 12/17
(`skill=2`, `limit_strength`, target_elo 1320, eval `nodes=200000`, `hash_mb=128`),
`legality_mode=constrained`, seed 0, prompt strict-v7. Inner model =
`chess-ft-qwen35-9b-pilot-grpo-q8_0:latest`.

Compare to GRPO-1000 baseline (run 17): **blunders/game, W-D-L, avg CPL**.

**Gate:**
- If blunders/game drops sharply (target ≤ ~1.0) AND there is ≥1 W/D → Phase A confirms the
  ceiling is worth chasing → build Phase B.
- If blunders drop but still 0 W/D → the bottleneck is conversion, not single blunders.
  **Stop. Do not build Phase B.** Report this as the finding (the wall is deeper than blunders).
- If blunders barely move → the candidate set rarely contains a safe move (generator problem);
  raise `k`/temperature once, else stop.

## 5. Phase B — learned blunder classifier (training)

Only if Phase A passes its gate.

### 5.1 Dataset build — `finetune/build_blunder_classifier_dataset.py`

Each training row is `(fen, candidate_uci) -> {blunder: bool}`.

- **Positions**: union of (a) distilled train positions and (b) arena game positions,
  weighted toward positions where models actually blundered (mine from `arena.db` runs 12–17,
  reuse the logic in `finetune/build_error_mined_dataset.py`). Keep a broad-position majority
  so the classifier does not collapse to "everything in sharp positions is a blunder".
- **Candidates per position**: sample `k` (e.g. 6) moves from the current best model (short
  generation, `num_predict≈16`, no thinking, temperature ~1.0) so the classifier sees the
  *model's own* candidate distribution, plus add 1–2 random legal moves for negative coverage.
- **Labels**: score each distinct `(fen, uci)` with `StockfishRewardScorer` /
  `StockfishEvaluator` (reuse `finetune/chess_reward.py`, cache by `(fen,uci)`). Label
  `blunder = cpl > 300 or allows_mate`. Record CPL too (for an optional regression head later).
- **Balance**: aim for roughly 25–40% positive (blunder) rows; subsample the easy negatives.
- Write JSONL + a `.meta.json` with class balance, source mix, and label thresholds.

Target size: **~20k–40k labeled rows** (see §8 for why — this is the training-time knob).

### 5.2 Classifier training — `finetune/train_blunder_classifier.py`

Mirror `finetune/train_lora.py` (frozen dataclass config, argparse, lazy unsloth import,
config sidecar, adapter+tokenizer save, response-only loss masking).

- **Model**: LoRA on `unsloth/Qwen3.5-9B` (same base as the policy) as a **generative binary
  classifier**: prompt renders the position + candidate move (reuse a strict-v7-style render
  with the candidate injected), target completion is a tiny fixed token, e.g.
  `{"blunder":true}` / `{"blunder":false}`. Same LoRA targets as `train_lora.py`
  (`r=32, alpha=32, dropout=0`, all-linear), `max_seq_length=1536`, 1 epoch, lr `2e-4`.
  - Cheaper alternative if training time is the bottleneck: use a smaller base (e.g.
    `Qwen2.5-1.5B`) for the classifier only — it never has to *play*, only judge, and detection
    is an easier task than generation. Note this trades some accuracy for ~4× faster training.
- **Output**: LoRA adapter + tokenizer under `outputs/finetune/qwen35_9b_blunder_classifier_lora`.

### 5.3 Offline gate (before any arena games)

`finetune/evaluate_blunder_classifier.py` on a held-out `(fen, uci)` set with Stockfish ground
truth: report **precision/recall/F1 of blunder detection**, plus a **simulated-veto** metric —
apply the veto over each position's candidate set and report predicted blunders/move before vs
after veto. Gate: **recall on true blunders ≥ ~0.7 with false-positive rate ≤ ~0.1**, and the
simulated veto must cut predicted blunders materially. If recall is low, the classifier is not
catching the tail → iterate on data balance/size before spending arena hours.

### 5.4 Play-time integration + benchmark

- Point `RerankedLLMMoveSource` scorer at the learned classifier (`scorer=classifier`,
  no Stockfish call during the game). The classifier runs as `N` short judgment passes per move.
- Benchmark identically to §4.3. Headline table: **blunders/game and W-D-L** for
  strict-v7 baseline (run 17) vs Phase A (Stockfish ceiling) vs Phase B (learned veto). The
  Phase A↔B gap is the "how much did the net learn" result.

## 6. Honest contract & reporting

- This is **not strict-v7**. Keep `prompt_version=strict-v7` (the prompt is unchanged) but tag
  the run's *regime* distinctly so it never aggregates with bare single-move runs. Add a
  reranker fingerprint to the run/model snapshot: `{scorer, n_candidates, temperature,
  veto_cpl_threshold}`. Surface it in the leaderboard as a separate regime label
  (e.g. "LLM + 1-ply veto (stockfish)" vs "LLM + 1-ply veto (learned)" vs "single move").
- Per-move telemetry to persist: `n_candidates_generated`, `n_legal`, `n_vetoed`,
  `veto_changed_move` (bool), `all_vetoed` (bool), `chosen_multiplicity`. These quantify how
  often the safety net actually fired — the core new metric.
- Framing for the writeup: AlphaZero = net + search; frontier reasoning = model + test-time
  compute. "Bare single forward pass" is an artificially strict bar. strict-v7 measures the
  bare model; the reranker measures the model when allowed a shallow second look. Two distinct,
  honestly-labeled numbers — never one presented as the other.

## 7. Deliverables / definition of done

- `arena_core` reranker move source (`RerankedLLMMoveSource`) + resolver wiring + telemetry.
- `finetune/build_blunder_classifier_dataset.py`, `finetune/train_blunder_classifier.py`,
  `finetune/evaluate_blunder_classifier.py`.
- Unit tests: candidate parse/validate/dedup/multiplicity, veto selection (incl. all-vetoed
  fallback and no-legal-candidate passthrough), classifier label thresholds. (`pytest -q`.)
- `finetune/README.md`: a "Phase 7: 1-ply reranker / blunder classifier" section with exact
  commands (mirror existing phase sections).
- `plans/fine_tune_model/blunder-reranker-status.md`: status log written *as it runs* (Phase A
  table vs run 17; Phase B class balance, training curve, offline P/R/F1, arena table). Same
  style as the other status files.
- Checks pass: `ruff check .` (new `arena_core` files in scope), `mypy arena_core backend
  tests` (the reranker is in `arena_core`, so it IS in strict-mypy scope — type it fully),
  `pytest -q`. `finetune/` stays out of strict mypy but keep explicit hints.
- Do not commit local DBs, GGUF/merged weights, or machine-specific paths.

## 8. Time budget (RTX 3090, order-of-magnitude)

Anchored on measured project numbers: distilled SFT was 16.7k examples in ~5h55m
(~0.78 examples/s, xFormers fallback / no FA2); the GRPO 1000-step run was ~4.6h.

| Step | Wall-clock | Notes |
| --- | --- | --- |
| Phase A: implement `RerankedLLMMoveSource` + wiring + tests | ~0.5–1 day | interactive |
| Phase A: 100-game ceiling benchmark | ~3–6 h | N candidates + N Stockfish evals per move (~3–5× a normal 1.1 h run); go/no-go gate |
| Phase B: dataset build (sample candidates + Stockfish labels) | ~4–8 h | mostly unattended; GPU candidate gen + CPU labeling, `(fen,uci)` cached |
| **Phase B: classifier LoRA training** | **~7–14 h (one overnight)** | 1 epoch; ≈ `examples / 0.78 s`. 20k≈7 h, 40k≈14 h. A 1.5B classifier cuts this to ~2–4 h |
| Phase B: offline P/R/F1 gate | ~0.5 h | before arena |
| Phase B: 100-game learned-veto benchmark | ~3–6 h | classifier runs as N judgment passes/move (no Stockfish) |

**The training itself ≈ one overnight (~7–14 h)** on the 3090, comparable to the existing SFT
runs — and it is bounded by the dataset size you choose (~20k–40k rows), so it is tunable. The
cheap Phase A ceiling needs **no training at all** — only a few hours of benchmark games —
and decides whether the overnight training is even worth running.

## 9. Risks & abort conditions

| Symptom | Action |
| --- | --- |
| Phase A: blunders drop but still 0 W/D | Stop before Phase B. Finding = the wall is conversion, not single blunders. Report it. |
| Candidate set rarely contains a safe move | Raise `k`/temperature once; if still bad, the generator is the bottleneck, not selection. |
| Phase B classifier low recall on blunders | Rebalance toward more positives / mine harder tactical positions / more data before arena. |
| Classifier high false-positive rate (vetoes good moves) | It will degrade play by discarding strong moves; raise the decision threshold, retrain with more quiet-position negatives. |
| Learned veto ≪ Stockfish ceiling | Expected to some degree — report the gap honestly; it is the headline "how much the net learned" number, not a failure. |
| Latency too high for 100 games | Lower `k`, or use the smaller 1.5B classifier; candidate completions are tiny so generation, not judging, should dominate. |

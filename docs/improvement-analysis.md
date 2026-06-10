# Improvement Analysis — LLM Chess Benchmark Arena

*Analysis date: 2026-06-10. Goal: portfolio PoC for software engineer / AI specialist roles,
focused on **model evaluation**, not a chess platform.*

## TL;DR

The project absolutely has potential as a local-LLM benchmark — the attempt-granular
persistence (every illegal/malformed move stored, not discarded) is genuinely the right
design and is what most hobby "LLM plays chess" projects lack. The engine and data model
are ~70% of a credible benchmark. What's missing is (1) **statistical rigor** so the numbers
are defensible, (2) **derived metrics** that tell a story (Elo estimate, accuracy %, failure-mode
taxonomy), and (3) **presentation** (README with results, charts, methodology writeup) — which
for a portfolio is the part recruiters actually see.

Separately: local models losing to even minimum-strength Stockfish is *expected*, and the
single biggest self-inflicted handicap in the current setup is the **UCI-only JSON contract**
(strict-v7) — models are pretrained on PGN/SAN text, not UCI. Details in section 5.

---

## 1. What is already strong (keep and showcase)

- **Attempt-granular persistence** — every LLM call stored with `parse_ok`, `legal_ok`,
  `error_type`, latency, tokens, thinking trace. This *is* the benchmark; say so loudly in the README.
- **Immutable `model_snapshot` keying** — leaderboard keyed on quantization/context/sampler
  fingerprint, not display name. This is the correct design and a great talking point
  (quantization effects on reasoning are an under-measured topic).
- **Clean architecture seam** — `arena_core` pure Python, backend thin, MoveSource protocol
  ready for the embodied phase. Easy to explain in an interview.
- **Legality modes as separate benchmarks** (open vs constrained) — illegal-move rate as a
  first-class metric is a real differentiator vs. win-rate-only benchmarks.
- **CI with ruff + mypy strict + pytest**, Alembic migrations, prompt versioning per attempt.

## 2. Benchmark credibility — fix before publishing numbers

These are the issues that would let a skeptical reviewer dismiss your results.

1. **Template hash doesn't hash the template.** `template_hash` is
   SHA256 of `"version:mode:legality_mode"` — change the template text without bumping the
   version and nothing detects it. Hash the rendered template skeleton text instead.
2. **Seeding is incomplete.** `random.seed()` is called once per tournament
   (`tournaments.py:71`); concurrent games or scheduling changes break reproducibility.
   Use a per-game `random.Random(seed + game_index)` instance, and add a determinism test:
   run the same config twice, assert identical games.
3. **No statistical reporting.** Add per-cell sample size to the leaderboard, Wilson
   confidence intervals on win/illegal rates, and a warning badge for n < 10 games.
   A benchmark without error bars is an anecdote.
4. **Opening coverage is 2 openings** (`STARTER_OPENINGS`). Use a small balanced suite
   (e.g. 10–20 positions from a standard opening book, both colors per opening) so results
   aren't dominated by one line.
5. **Snapshot fingerprint gaps.** `sampler_params` is frozen to a constant dict in
   `tournaments.py` while actual per-run knobs (num_ctx, num_predict, num_gpu) vary — store
   the real values. `ollama_digest` silently falls back to None when `/api/show` fails.
6. **CLAUDE.md / docs drift.** Docs say `strict-v6` + SAN-with-UCI-fallback; code is
   `strict-v7` + UCI-only (`engine._parse_uci_move` only tries `Move.from_uci`). For a
   benchmark, the documented contract *is* the methodology — keep them in sync.
7. **Dead fields look unfinished:** `cost_usd` always 0.0, `truncation_applied` always False.
   Either implement or remove — half-populated columns undermine trust in the rest.

## 3. Missing metrics (high value, mostly cheap to add)

All raw data already exists in `moves`/`attempts`/`engine_evaluations`; these are
aggregation-layer additions (`leaderboards.py` + one Alembic migration + UI columns).

| Metric | Why it matters |
|---|---|
| **Estimated Elo** | The headline number everyone understands. Play each model vs Stockfish at several `UCI_Elo` levels (1320, 1500, 1700…) and fit a logistic curve — even a crude anchor ("plays ~600 Elo") is far more communicative than avg CPL. |
| **Accuracy %** = (best+good)/moves | `best`/`good` classifications exist but are never aggregated — only blunders/mistakes/inaccuracies are counted today. |
| **CPL distribution / percentiles** | Mean CPL hides whether a model is "consistently mediocre" or "great until one blunder". Store a small histogram (bins matching classification thresholds). |
| **Game-phase breakdown** | Illegal-move rate and CPL split by opening/middlegame/endgame (ply buckets). Your Gemma observation ("loses track of position") will show up as illegal rate climbing with ply — that's a *publishable chart*. |
| **Error-type taxonomy** | % malformed JSON vs % syntactically-valid-but-illegal vs % illegal-after-feedback. This is the "model behavior differences" story you already see qualitatively. |
| **Retry distribution** | % of moves needing 0/1/2/3+ retries, not just the mean. |
| **Latency p50/p95** and thinking-token share | Already captured per attempt; aggregate them. |
| **Hallucinated-piece rate** | For illegal moves: classify *why* illegal (no piece on from-square / blocked / leaves king in check / bad syntax). Cheap with python-chess, and it directly measures board-state tracking — the failure mode you observed. |

## 4. Portfolio presentation (highest ROI for the job-search goal)

The engine is good; the bottleneck is that nobody can *see* it's good in 60 seconds.

1. **README rewrite** — a recruiter spends ~1 minute:
   - One screenshot of the leaderboard + one GIF of game replay with the eval graph.
   - A real results table: 4–6 local models (include a Gemma so the failure modes show)
     vs random + Stockfish skill 0, with illegal-rate and accuracy columns.
   - A short **Methodology** section: why attempt-level logging, what CPL is, why
     immutable snapshots, why legality modes are separate. This section *is* the
     AI-specialist signal.
   - Limitations + roadmap (Phase 2 embodied board) — showing you know the limits reads
     as seniority.
2. **Ship demo data** — a committed `demo/arena-demo.db` (or a seed script) so
   `make demo` brings the UI up with real results. Empty-state portfolio apps die in review.
3. **Frontend: split `App.tsx` (2,509 lines)** into `GameReplay`, `Leaderboard`,
   `RunComparison`, `DebugPanel` components. This is the file an interviewer will open.
4. **Add 2–3 benchmark charts** (Recharts is already in): illegal-rate vs ply, CPL
   distribution per model, win-rate matrix (head-to-head grid). Sortable leaderboard columns.
5. **Run detail view** — `GET /runs/{run_id}` (config, seed, stockfish version, prompt
   version, participants) + UI panel. Reproducibility metadata on screen is a benchmark-credibility flex.
6. **PGN export endpoint** — the PGN is already stored on `games`; expose it so results
   can be checked in any chess GUI.
7. **A short writeup** (`docs/results.md` or a blog post linked from README) analyzing one
   interesting finding — e.g. "Gemma-2 9B's illegal-move rate triples after move 20" or
   "Q4 vs Q8 quantization costs X accuracy points". One real insight beats ten features.

## 5. Why local LLMs lose to Stockfish — and how to make them better

First, calibrate expectations: **even Stockfish skill 0 plays roughly 1300+ Elo**, while
small chat-tuned local models play far below beginner level. GPT-3.5-turbo-instruct famously
reached ~1750 Elo *only* in raw PGN-completion mode; chat models given the same task play
hundreds of points worse. Losing to minimum Stockfish is the expected baseline, not a bug
in your harness. That's also why the benchmark's value is *relative comparison + failure
modes*, not "can it beat Stockfish".

Concrete levers, roughly in order of expected impact:

1. **Switch the contract back to SAN (or test both).** Models are pretrained on millions of
   PGN games written in SAN (`Nf3`, `exd5`); UCI (`g1f3`) is nearly absent from training
   data. The current strict-v7 UCI-only contract forces models to translate into a notation
   they barely know — you're benchmarking notation translation, not chess. Make
   notation a benchmark dimension: `strict-v8-san` vs `strict-v8-uci` on identical games.
2. **Add a PGN-completion prompt mode.** Instead of JSON chat, present the game as a PGN
   header + movetext ending mid-game and let the model complete the next SAN token
   (low temperature, stop at whitespace). This is the single best-known trick for LLM chess
   strength. Keep it as a separate prompt version so it's comparable — it also makes a
   great experiment chapter: "chat-JSON vs PGN-completion cost N hundred Elo".
3. **Weaker/graded baselines so games are informative.** Add anchor opponents between
   random and Stockfish: Stockfish with `UCI_LimitStrength` + low Elo *and* 1-node search,
   a greedy material-capture bot, and ideally a Maia-style human-like net. A ladder of
   anchors is what turns win-rates into an Elo estimate.
4. **Constrained decoding (for the constrained mode).** Ollama supports structured
   outputs (JSON schema); you can supply `{"move": {"enum": [<legal SAN list>]}}` per turn
   so the model *cannot* emit an illegal move. Keep it as a third legality mode
   (`grammar`) — comparing open vs constrained vs grammar isolates "doesn't know rules"
   from "can't follow format".
5. **Best-of-N self-consistency.** Sample N=3–5 proposals at temperature ~0.7 and majority-vote
   (or take the first legal one). Cheap, measurable accuracy gain, and "samples vs accuracy"
   is itself a nice benchmark curve.
6. **Prompt-content ablations** (you already have the versioning machinery):
   - ASCII board on/off (it's hardcoded on; for some models it may *hurt* — measure it).
   - FEN vs board diagram vs move-list-only — directly tests the state-tracking failure
     you saw in Gemma.
   - "List all your legal-looking candidate moves, then pick one" (forced verification step).
   - Few-shot example of the exact expected output.
7. **Fine-tuning (the AI-specialist showcase).** LoRA-tune a small model (Qwen 3 4B /
   Llama 3.2 3B) on Lichess games formatted exactly like your prompt, then benchmark
   base vs tuned *in your own arena*. "I built the benchmark, identified the weakness,
   fine-tuned, and measured +N00 Elo / −80% illegal rate" is the strongest possible
   portfolio narrative — the arena becomes the eval harness for your own training work.
8. **Don't add Stockfish-assisted move selection to scored runs** (e.g. "pick from engine
   top-3") — it contaminates the benchmark. If you want a playable "centaur" mode, keep it
   outside benchmark runs, consistent with the no-silent-substitution invariant.

## 6. Code-quality cleanups (lower priority, do opportunistically)

- **Parser** (`parser.py`): first-balanced-`{}` extraction grabs the wrong object when the
  model emits multiple JSON blocks; broken escapes can derail depth tracking. Add tests for
  multi-object responses, nested `"move"` keys, and giant prose preambles.
- **Backend**: `_run_game_job` / `_run_stockfish_match_job` share 10+ duplicated params —
  extract a config dataclass. Job state is in-process memory (lost on restart) — fine for a
  PoC, but persist job rows if you demo it. `/health` should check DB + Stockfish.
- **LLM clients**: a new httpx client per source; pool/reuse for long tournaments.
- **Tests**: add determinism test (same seed twice → identical games), classification
  boundary tests (20 vs 21 cp), migration-from-scratch test, and a coverage report in CI.
  Frontend has zero tests — even one Vitest smoke test of leaderboard rendering helps.
- **CI**: add `pytest --cov` with a badge, `tsc -b` type-check step for the frontend.
- **Hygiene**: `.gitignore` the `.playwright-mcp/` snapshots; move `plans/*.md` into `docs/`.

## 7. Suggested order of work

1. **Credibility fixes** (template content hashing, per-game seeds, real sampler_params,
   sample-size + CI on leaderboard) — small, do first so every later run is citable.
2. **Metrics layer** (accuracy %, Elo-vs-Stockfish ladder, phase breakdown, error taxonomy).
3. **One big benchmark run** across 5–6 local models with the new metrics → demo DB.
4. **Presentation** (README + screenshots + results writeup + App.tsx split + charts).
5. **The experiment** (SAN vs UCI, PGN-completion mode, optionally the LoRA fine-tune) —
   this is the chapter that turns "nice project" into "AI specialist portfolio".

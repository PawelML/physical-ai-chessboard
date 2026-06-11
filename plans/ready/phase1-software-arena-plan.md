# Phase 1 — Software-Only LLM Chess Arena (Implementation Plan)

> This is the detailed plan for the **software-only first phase** of the larger
> [Embodied AI Chess Lab](embodied-ai-chess-lab-plan.md). No camera, no robot yet.
> The goal of this phase is a **rigorous, reproducible benchmark** in which LLMs
> (local and, later, API) play chess, and every move, failure, retry, cost,
> latency and engine evaluation is persisted and queryable.

## 1. Core thesis

This is **not "an app where LLMs play chess"**. It is an **experiment runner**.
The portfolio value is that the system is *reproducible and queryable*: prompt
version, model snapshot, sampling config, Stockfish version, opening line and the
full attempt history (including illegal-move failures) are all captured. Chess is
the **benchmark substrate** that discriminates model strength and rule-following.

Companion design docs:
- [`docs/architecture.md`](../docs/architecture.md) — `arena-core` package + backend + frontend
- [`docs/data-model.md`](../docs/data-model.md) — full persistence schema
- [`docs/move-loop-and-prompts.md`](../docs/move-loop-and-prompts.md) — prompt contracts, two modes, retry protocol
- [`docs/evals.md`](../docs/evals.md) — metrics, tournaments, leaderboard, determinism

## 2. Locked decisions

| Decision | Choice |
| --- | --- |
| v1 deliverable | Full-stack (`arena-core` + FastAPI + React), built in visible-early phases |
| Quality judge | **Stockfish from the start** — fixed nodes, single thread, pinned version |
| Move modes | **strict** (move-only JSON, scored) + **reasoning/persona** (annotation, *not* scored) |
| Legality modes | **open** (free proposal) + **constrained** (given legal move list) — separate leaderboards |
| Persistence | **SQLite** on start, schema written Postgres-ready (SQLAlchemy + Alembic) |
| Opponents | **model-vs-model** and **model-vs-Stockfish** (skill-capped) |
| Openings | **Versioned opening suite from the start**, every line played with both colors |
| v1 models | **Local only** (Ollama). API providers (Anthropic/Gemini/OpenAI) supported in the abstraction, enabled behind a flag later |

### Local model inventory (Ollama, RTX 3090 24 GB)

| Model | Footprint | Concurrent play? |
| --- | --- | --- |
| `qwen3.5:9b` | 6.6 GB | yes |
| `gemma3n:e4b` | ~5 GB (`ollama pull gemma3n:e4b` — **not yet installed**) | yes |
| `qwen3-vl:8b` | 6.1 GB | yes |
| `nanbeige4.1:3b` | 4.2 GB | yes |
| `qwen3.5-27b-ud` | 17 GB | solo, or swap-per-turn |
| `qwen3.5:35b-a3b` | 23 GB | **solo only** (swaps each turn if paired) |
| `nemotron-3-nano` | 24 GB | **solo only** |

First test tournament: **`qwen3.5:9b` vs `gemma3n:e4b`** (both fit resident, fast).
The arena computes the pair's VRAM footprint and, when it exceeds budget, warns and
records **model-swap latency** as an operational metric rather than failing.

## 3. Hard rules (source of truth)

- **`python-chess` owns game state.** The LLM never owns or mutates state.
- **Every failed attempt is persisted.** Illegal/malformed moves are first-class
  benchmark signal, not noise.
- **Leaderboard rows are keyed by an immutable model *snapshot*** (Ollama digest,
  quantization, context window, sampler params), never by display name.
- **strict and reasoning never share a score.** Commentary annotates an
  already-scored strict move; it never enters the scoring path.
- **open and constrained are separate leaderboards.** Combining them is meaningless.

## 4. Phase breakdown (revised — visible-early, benchmark-credible-early)

Each phase is independently demoable. Ordering follows the principle: *get to a
credible scored replay in the browser as early as possible*, then scale to
tournaments and leaderboards.

### Phase 0 — Scaffold
- Repo layout (`arena-core/`, `backend/`, `frontend/`), `pyproject.toml`, ruff/mypy.
- Pydantic Settings (ported pattern from `digit_assistant`).
- SQLite + SQLAlchemy 2.0 async + **Alembic migrations from day one**.
- pytest harness with `unit` / `integration` markers; CI (lint + unit tests).
- Smoke test: a **fake/random agent** plays one legal move end-to-end and persists it.
- **Demo:** `pytest` green; one move written to the DB.

### Phase 1 — `arena-core` engine MVP (+ basic telemetry baked in)
- python-chess game loop; turn handling; termination detection; PGN export.
- **strict-mode** prompt builder (FEN + SAN history + the model's own move list +
  last opponent move + optional ASCII board). See move-loop doc.
- UCI/JSON parsing; legality validation; **bounded retry (max 3)** with structured
  feedback (`error`, `attempted_move`, `reason`, `legal_moves`, `remaining_retries`).
- Persistence: `games`, `moves`, `attempts`.
- **Basic telemetry merged here** (not a separate later phase): per-attempt latency,
  prompt/response token estimate, retry count, context-length estimate.
- CLI: run one game `<model> vs <model|random>` and print the PGN + an attempt log.
- **Demo:** a local model plays a full game from the CLI; a PGN and per-attempt rows exist.

### Phase 2 — Stockfish evaluator (moved up — this is what makes it a *benchmark*)
- Pinned Stockfish binary/version; single thread; fixed nodes; explicit UCI options.
- Eval **before and after** each accepted move → centipawn loss (CPL).
- **Mate scores stored separately** (never naively converted to centipawns);
  explicit classification rules for mate positions.
- Blunder / mistake / inaccuracy classification; store `best_move`, nodes, depth reached.
- Persistence: `engine_evaluations`.
- **Demo:** "model played a game — here are its blunders, with CPL per move."

### Phase 3 — Minimal backend + thin replay UI (visible portfolio demo early)
- FastAPI: list runs/games, fetch one game with moves + evals.
- React: board **replay** (step through plies), per-move panel (CPL / legality /
  retries / latency / tokens). **Polling, no SSE yet.**
- **Demo:** open a browser, replay a real game, see real CPL/blunder annotations.

### Phase 4 — Tournament runner
- model-vs-model and model-vs-Stockfish (skill-capped) formats.
- Opening suite (versioned), every line played with **both colors**.
- `benchmark_runs`, `run_participants`, `opening_suites`, `opening_lines` entities.
- **Deterministic config capture + config hash:** temperature=0, sampler params,
  prompt version, model snapshot/digest, Stockfish version+options, git commit, seed.
- **Demo:** run a small tournament; multiple games persisted under one run.

### Phase 5 — Rich telemetry + leaderboard aggregation
- Context growth tracking; retry analytics; **model-swap latency** + VRAM-pair warnings.
- Cost fields populated (nullable; local = 0) — schema present, no UI polish yet.
- Materialized `game_summaries` / leaderboard aggregates (CPL, blunders, illegal
  rate, retries, W/D/L by model snapshot / run / opening / color / mode).
- **Demo:** a queryable leaderboard table (separate per legality mode).

### Phase 6 — Full React UI
- Leaderboard with all filter dimensions (see evals doc); run comparison.
- Live **SSE** stream of in-progress games (now that multi-game runs exist).
- Richer replay (eval graph over plies, attempt drill-down).
- **Demo:** the polished portfolio surface.

### Phase 7 — Reasoning/persona + reports
- reasoning/persona mode (aggressive / positional / defensive / risk-taking /
  technician) — commentary as **annotation over an already-scored strict move**,
  explicitly outside scoring.
- Match report export (PGN + stats + commentary).
- **Demo:** a generated match report with personas and commentary.

## 5. What is intentionally cut / deferred from v1

- Real cost accounting polish (local models are free; keep nullable fields).
- SSE before tournaments exist (polling is fine for the first UI).
- API providers enabled by default (abstraction ready; flag-gated, enabled later).
- Anything from the robot/perception phases (out of scope until Phase 1 ships).

## 6. Reused from `digit_assistant`

- `BaseLLMService` (ABC) + Gemini / Anthropic / Local(OpenAI-compatible) →
  normalized `LLMResponse`. Ollama plugs into the local adapter directly.
- `token_counter.py` (tiktoken) and `context_manager.py` (truncation).
- Pydantic Settings config pattern; Prometheus metrics.
- React 19 + Vite + TanStack Query + shadcn/ui; pytest unit/integration markers.

## 7. Determinism contract (must hold for the benchmark to be credible)

- `temperature=0` is **not** full determinism for local LLMs — also capture seed
  (if the runtime supports it), Ollama version, model digest, quantization, sampler
  params, context size.
- Pin Stockfish binary/version and options; **single thread**; store actual nodes
  used and depth reached.
- Version the opening suite; report per-opening **and** aggregate; both colors.
- model-vs-Stockfish skill capping is not deterministic unless tightly configured —
  capture skill / UCI_LimitStrength / Elo / nodes / seed.

See [`docs/evals.md`](../docs/evals.md) for the full determinism and metrics treatment.

## 8. Immediate next actions (Phase 0 kickoff)

1. `git init`; create repo skeleton per [`docs/architecture.md`](../docs/architecture.md).
2. `ollama pull gemma3n:e4b` (verify footprint with `ollama ps`).
3. Stand up Settings + SQLite + SQLAlchemy + first Alembic migration.
4. Random-agent smoke test (one legal move persisted) green in CI.

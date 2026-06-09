# Architecture — Software Arena (Phase 1)

> See the implementation plan in
> [`plans/phase1-software-arena-plan.md`](../plans/phase1-software-arena-plan.md).

## Design principle

A clean, web-free Python package (`arena-core`) is the heart of the system.
FastAPI is a **thin shell** over it; React is a presentation layer. This keeps the
benchmark engine reusable for the later robot phase, where the physical board simply
swaps in a different *move source* and *board-state observer* without dragging web
code into robotics.

```text
arena-core   (pure Python, no web, no FastAPI imports)
   ├── engine        python-chess game loop = sole source of truth for state
   ├── llm           BaseLLMService (ABC) + Local/Anthropic/Gemini/OpenAI adapters
   ├── prompts       versioned prompt templates (strict / reasoning × open / constrained)
   ├── evaluators    Stockfish wrapper (fixed nodes, pinned version) → CPL, classification
   ├── tournaments   model-vs-model, model-vs-Stockfish, opening suites, run config
   ├── persistence   SQLAlchemy models + repositories (SQLite now, Postgres-ready)
   ├── telemetry     token counting, context tracking, latency, VRAM/swap detection
   └── config        Pydantic Settings

backend      (thin FastAPI)
   ├── api           create/list runs, fetch games+moves+evals, leaderboard queries,
   │                 game-job runner, GET/PUT /settings/game-defaults (sampling knobs)
   ├── stream        SSE of live game progress (Phase 6; polling before that)
   └── db            session wiring, migrations entrypoint

frontend     (React 19 + Vite + TanStack Query + shadcn/ui)
   ├── replay        step through a game, per-move CPL/legality/latency/token panel
   ├── leaderboard   filterable table (Phase 5–6)
   └── live          SSE live board (Phase 6)
```

## `arena-core` components

### engine
- Wraps `python-chess`. Owns `board`, legal-move generation, SAN/UCI conversion,
  termination/result detection, PGN export.
- Exposes a `Game` object that drives the move loop and emits structured events
  (move accepted, attempt failed, game ended) for persistence and streaming.
- A **MoveSource** abstraction (LLM agent, random baseline, Stockfish, and — in the
  robot phase — human/robot/camera) feeds moves into the loop. This is the seam the
  later embodied phase reuses.

### llm
- Ported `BaseLLMService` (ABC) returning a normalized `LLMResponse`
  (`content`, `tool_calls`, `stop_reason`, `raw_response`).
- Adapters: `LocalLLMService` (OpenAI-compatible → Ollama), `AnthropicLLMService`,
  `GeminiLLMService`, `OpenAILLMService`. v1 uses local only; others are flag-gated.
- Each adapter reports usage (prompt/completion tokens when available) and latency
  back to `telemetry`.
- Ollama adapter captures the model thinking trace and whether thinking actually ran
  (silently disabled for unsupported builds — see move-loop doc). A cold load of a
  large model can exceed 100s, so `ollama_timeout_seconds` defaults to 600 and a
  timeout surfaces a clear `OllamaServiceError` rather than an empty failure.

### prompts
- Versioned templates. A `prompt_version` is stored on every attempt so results are
  reproducible. Four template families: `{strict, reasoning} × {open, constrained}`.
- The strict prompt states the win objective and asks for a move in **UCI**
  (`{"move":"e2e4"}`); the engine accepts only UCI. Current version `strict-v7`.
- See [`move-loop-and-prompts.md`](move-loop-and-prompts.md).

### evaluators
- Stockfish subprocess via `python-chess` UCI. Pinned version, single thread, fixed
  nodes, explicit options. Returns eval-before/after, best move, CPL, mate flag,
  classification. Mate scores kept separate from centipawns.

### tournaments
- Builds pairings (both colors), iterates opening lines, runs games, captures the
  full deterministic config + config hash, writes `benchmark_runs` + child rows.

### persistence
- SQLAlchemy 2.0 async models + repositories. SQLite file now; the schema avoids
  SQLite-only constructs so a Postgres switch is a connection-string change plus
  Alembic. Attempt-level granularity (see [`data-model.md`](data-model.md)).

### telemetry
- tiktoken-based token estimate (treated as approximate across heterogeneous local
  models, never as exact), context-window usage/remaining, per-attempt latency, and
  VRAM-pair footprint / model-swap-latency detection for oversized Ollama pairs.

## Boundaries

- `arena-core` imports **no** web framework. The backend depends on `arena-core`,
  never the reverse.
- The frontend talks only to the backend HTTP/SSE API, never to `arena-core` or the DB.
- Scoring logic lives entirely in `arena-core/evaluators` + `tournaments`; the UI
  only reads aggregates.

## Tech stack (carried from `digit_assistant`)

- Backend: Python 3.12+, FastAPI, SQLAlchemy 2.0 async, Alembic, Pydantic Settings,
  Prometheus, pytest (`unit`/`integration` markers).
- Frontend: React 19, Vite, TypeScript, TanStack Query, shadcn/ui (Radix + Tailwind),
  Recharts for the eval graph.
- Local inference: Ollama (OpenAI-compatible endpoint).
- Engine: Stockfish (pinned binary), `python-chess`.

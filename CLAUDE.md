# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Phase 1 of the Physical AI Chessboard: a **software-only LLM chess benchmark arena**.
`python-chess` owns all game state; LLMs only propose moves. Every move attempt —
including illegal/malformed failures — is persisted at attempt granularity, because
that failure data *is* the benchmark. A later "embodied" phase will reuse the engine
by swapping in a physical-board move source; keep that seam clean.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # package + CLI + backend + dev tools

arena init-db                                                   # create tables
arena play random random --db-url sqlite+aiosqlite:///./arena.db
arena play <ollama-model> random --legality-mode constrained --stockfish-path <path>
arena tournament <a> <b> --name run1 --seed 0                   # both-colors pairing + summaries
arena rebuild-summaries [--run-id N]                            # rematerialize game_summaries
arena export-report <game_id> [-o out.md]
arena show-db                                                   # resolve configured DB path

# Backend (thin FastAPI over arena_core)
ARENA_DATABASE_URL=sqlite+aiosqlite:///./arena.db uvicorn backend.main:app --reload

# Frontend (React 19 + Vite replay/leaderboard UI)
cd frontend && npm install && npm run dev

# Local Stockfish without sudo
bash scripts/install_stockfish.sh
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
```

### Checks (match CI in `.github/workflows/ci.yml`)

```bash
ruff check .                       # rules E,F,I,UP,B,ASYNC; 100-char lines
mypy arena_core backend tests      # strict mode
pytest -q                          # asyncio_mode=auto; markers: unit, integration
pytest tests/test_game_loop.py::test_name   # single test
cd frontend && npm run lint && npm run build
```

Alembic migrations live in `alembic/versions/`; the env reads `-x db_url=...` or
`sqlalchemy.url`. `init-db` calls `create_tables` directly (used in dev/tests);
migrations are the schema-of-record for upgrades.

## Architecture

`arena_core/` is a **pure-Python package with no web framework imports**. The
backend depends on it; never the reverse. The frontend talks only to the backend
HTTP/SSE API. This boundary is load-bearing — don't import FastAPI into `arena_core`.

```
arena_core/
  engine.py        ArenaGame move loop; MoveSource Protocol (the embodied-phase seam):
                   RandomMoveSource, StaticMoveSource, LLMMoveSource, StockfishMoveSource
  prompts.py       versioned templates, {strict,reasoning} × {open,constrained} legality modes
  parser.py        extract UCI move from raw model JSON
  move_sources.py  source resolution helpers
  llm/             LLMService ABC -> LLMResponse; OpenAICompatible(Ollama), Anthropic, Gemini
  evaluators/      Stockfish UCI wrapper (fixed nodes, pinned binary) -> CPL, classification
  tournaments.py   pairings, run config + config_hash, writes benchmark_runs + children
  leaderboards.py  rebuild_game_summaries (materialized aggregates)
  annotations.py   persona commentary over already-scored moves (NOT a scoring path)
  reports.py       per-game Markdown export
  telemetry.py     approximate token counts, context usage, latency, VRAM/offload detection
  persistence/     SQLAlchemy 2.0 async models + repositories (SQLite now, Postgres-ready)
  config.py        pydantic-settings, env prefix ARENA_
backend/main.py    create_app(): game-job runner, runs/games/leaderboard queries, SSE stream
frontend/src/      App.tsx, api.ts, chess.ts (TanStack Query)
```

## Invariants that are easy to break

- **`python-chess` is the sole source of truth.** The model never owns or mutates
  state. The prompt is rebuilt from game state every turn (not accumulated as chat
  history); the model's own prior moves are passed in explicitly — that is the
  "memory" requirement, made deterministic.
- **Never silently substitute a move in a scored run.** On retry exhaustion (default
  `max_retries=3`), record `forfeit_invalid` as the termination. Failed attempts are
  persisted, not discarded.
- **Legality modes are separate benchmarks**, always reported separately: `open`
  (no legal list given; illegal-move rate is a first-class metric) vs `constrained`
  (legal moves provided). On any *retry*, legal moves are always provided regardless
  of mode.
- **Prompt versioning is mandatory.** Every template change bumps `prompt_version`
  (currently `strict-v4` in `config.py`) and `template_hash`, both stored per attempt.
  Comparisons across prompt versions must be explicit.
- **Leaderboard rows key on an immutable `model_snapshot`**, never a display name —
  the snapshot fingerprints quantization, context window, sampler params, runtime
  version. Aggregates live in `game_summaries`; rebuild via `rebuild-summaries` after
  changing scoring. The UI only reads aggregates; scoring lives in
  `evaluators` + `tournaments`.
- **Token counts are approximate** across heterogeneous local models — never present
  as exact. Stockfish mate scores are kept separate from centipawns.

## Move sources & local models

`_source_from_name` (in `cli.py`) maps a name to a `MoveSource`: `"random"`,
`"stockfish"` (needs `ARENA_STOCKFISH_PATH`/`--stockfish-path`), or otherwise an
Ollama model name via `LLMMoveSource`. API providers (Anthropic/Gemini/OpenAI) are
flag-gated behind `ARENA_API_PROVIDERS_ENABLED`; v1 runs local-only. Ollama runtime
knobs (temperature, num_ctx, num_gpu, CPU-offload layers, think mode, VRAM budget)
are all `ARENA_OLLAMA_*` settings — see `config.py`. The backend exposes presets
(`strict`/`low_creativity`) and guidance modes (`legal_list`/`strategic_memory`).

## Conventions

Python 3.12, snake_case functions/modules, PascalCase for Pydantic/SQLAlchemy models.
Keep type hints explicit (mypy strict). Frontend is TypeScript/React: PascalCase
components, camelCase functions/state. Commits are short imperative summaries scoped
to one behavior; note DB/prompt-version changes in PRs. Do not commit local DBs,
model outputs, secrets, or machine-specific paths.

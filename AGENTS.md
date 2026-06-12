# Repository Guidelines

## Project Structure & Module Organization

This repository contains a software LLM chess arena. Core Python logic lives in `arena_core/`: game loop, prompts, parsers, telemetry, persistence, reports, leaderboards, tournaments, Stockfish evaluation, and LLM providers. Fine-tuning smoke and training utilities live in `finetune/`; fine-tuning plans and progress notes live in `plans/fine_tune_model/`. The FastAPI backend is in `backend/`, with Alembic migrations in `alembic/`. The React/Vite replay and benchmark UI is in `frontend/src/`. Tests are in `tests/`, docs in `docs/`, helper scripts in `scripts/`, and vendored local tools can live under `vendor/`.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create the Python environment.
- `pip install -e ".[dev]"`: install the package, CLI, backend extras, and dev tools.
- `python -m venv .venv-train && source .venv-train/bin/activate && pip install -e ".[train]"`: create the separate fine-tuning environment.
- `arena init-db`: initialize the configured arena database.
- `alembic -x db_url=sqlite+aiosqlite:///./arena.db upgrade head`: apply database migrations to an existing local database.
- `arena play random random --db-url sqlite+aiosqlite:///./arena.db`: run a sample game.
- `arena rebuild-summaries --db-url sqlite+aiosqlite:///./arena.db`: rebuild materialized benchmark summaries after summary schema or scoring changes.
- `bash scripts/install_stockfish.sh`: install the pinned local Stockfish package under `vendor/stockfish/` without sudo.
- `ARENA_DATABASE_URL=sqlite+aiosqlite:///./arena.db uvicorn backend.main:app --reload`: run the API locally.
- `cd frontend && npm install && npm run dev`: start the Vite UI.
- Preferred local app URL for this machine: `http://192.168.10.73:5173/`.
  Start Vite on `0.0.0.0`/port `5173` so the UI is reachable from the LAN.
- `pytest`: run Python tests.
- `ruff check backend arena_core tests`: lint Python code.
- `ruff check backend arena_core finetune tests`: lint Python code including fine-tuning utilities.
- `python -m finetune.build_smoke_dataset --pgn <file.pgn> --output data/finetune/examples.jsonl --metadata-output data/finetune/examples.meta.json`: build strict-v7 fine-tuning examples from PGN.
- `python -m finetune.smoke_train --dataset data/finetune/examples.jsonl --output-dir outputs/finetune/<run> --max-steps 100`: run a small QLoRA smoke train from `.venv-train`.
- `cd frontend && npm run lint && npm run build`: lint and build the frontend.

## Coding Style & Naming Conventions

Python targets 3.12, uses 100-character lines, and is linted with Ruff rules `E`, `F`, `I`, `UP`, `B`, and `ASYNC`. Keep type hints explicit; MyPy is strict. Use `snake_case` for Python functions/modules and `PascalCase` for Pydantic or SQLAlchemy models. Fine-tuning examples must be rendered by `arena_core.prompts.build_strict_prompt`, and completion targets must stay exactly `{"move":"<uci>"}`. Frontend code is TypeScript/React; use `PascalCase` for components, `camelCase` for functions and state, and colocate local UI helpers in `frontend/src/`.

## Testing Guidelines

Use Pytest for backend and core tests. Name test files `test_*.py` and keep tests close to the behavior changed, especially move parsing, prompts, retries, telemetry, persistence, Stockfish integration, tournaments, summary scoring, and fine-tuning dataset generation. Mark tests with `unit` or `integration` when useful. For frontend changes, run `npm run lint` and `npm run build`; add focused tests only if a frontend test harness is introduced.

## Commit & Pull Request Guidelines

Commit history uses short, imperative summaries such as `Add strategic memory guidance for games` and `Fix benchmark scoring and harden arena engine`. Keep commits scoped to one behavior. Pull requests should describe the user-visible change, note database migrations or prompt-version changes, list validation commands, and include screenshots or recordings for UI changes.

## Security & Configuration Tips

Do not commit local databases, generated fine-tuning datasets, LoRA adapters, GGUF files, model outputs, secrets, or machine-specific paths. Keep generated fine-tuning artifacts under ignored `data/finetune/` and `outputs/finetune/`. Configure runtime values with environment variables such as `ARENA_DATABASE_URL` and `ARENA_STOCKFISH_PATH`; the vendored fallback path is `vendor/stockfish/root/usr/games/stockfish`. Ollama games depend on local models and runtime options, so document model names, thinking mode, context size, Stockfish level, game count, and presets when sharing benchmark results. Fine-tuning progress should be tracked in `plans/fine_tune_model/`; Phase 0 status is in `plans/fine_tune_model/phase0-status.md`.

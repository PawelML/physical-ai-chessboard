# Repository Guidelines

## Project Structure & Module Organization

This repository contains a software LLM chess arena. Core Python logic lives in `arena_core/`: game loop, prompts, parsers, telemetry, persistence, reports, and LLM providers. The FastAPI backend is in `backend/`, with migrations in `alembic/`. The React/Vite replay UI is in `frontend/src/`. Tests are in `tests/`, docs in `docs/`, plans in `plans/`, and helper scripts in `scripts/`.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate`: create the Python environment.
- `pip install -e ".[dev]"`: install the package, CLI, backend extras, and dev tools.
- `arena init-db`: initialize the configured arena database.
- `arena play random random --db-url sqlite+aiosqlite:///./arena.db`: run a sample game.
- `ARENA_DATABASE_URL=sqlite+aiosqlite:///./arena.db uvicorn backend.main:app --reload`: run the API locally.
- `cd frontend && npm install && npm run dev`: start the Vite UI.
- `pytest`: run Python tests.
- `ruff check backend arena_core tests`: lint Python code.
- `cd frontend && npm run lint && npm run build`: lint and build the frontend.

## Coding Style & Naming Conventions

Python targets 3.12, uses 100-character lines, and is linted with Ruff rules `E`, `F`, `I`, `UP`, `B`, and `ASYNC`. Keep type hints explicit; MyPy is strict. Use `snake_case` for Python functions/modules and `PascalCase` for Pydantic or SQLAlchemy models. Frontend code is TypeScript/React; use `PascalCase` for components, `camelCase` for functions and state, and colocate local UI helpers in `frontend/src/`.

## Testing Guidelines

Use Pytest for backend and core tests. Name test files `test_*.py` and keep tests close to the behavior changed, especially move parsing, prompts, retries, telemetry, and persistence. Mark tests with `unit` or `integration` when useful. For frontend changes, run `npm run lint` and `npm run build`; add focused tests only if a frontend test harness is introduced.

## Commit & Pull Request Guidelines

Commit history uses short, imperative summaries such as `Add strategic memory guidance for games` and `Fix benchmark scoring and harden arena engine`. Keep commits scoped to one behavior. Pull requests should describe the user-visible change, note database or prompt-version changes, list validation commands, and include screenshots or recordings for UI changes.

## Security & Configuration Tips

Do not commit local databases, model outputs, secrets, or machine-specific paths. Configure runtime values with environment variables such as `ARENA_DATABASE_URL` and `ARENA_STOCKFISH_PATH`. Ollama games depend on local models, so document model names and presets when sharing results.

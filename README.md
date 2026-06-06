# Physical AI Chessboard

Phase 1 is a software-only LLM chess arena: `python-chess` owns game state, move
attempts are persisted at attempt granularity, and the CLI can run reproducible
strict-mode games against random or local Ollama-backed agents.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
arena init-db
arena play random random --db-url sqlite+aiosqlite:///./arena.db
```

Install the pinned local Stockfish package without sudo:

```bash
bash scripts/install_stockfish.sh
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
```

Run the backend:

```bash
ARENA_DATABASE_URL=sqlite+aiosqlite:///./arena.db uvicorn backend.main:app --reload
```

Run the frontend replay UI:

```bash
cd frontend
npm install
npm run dev
```

Run tests:

```bash
pytest
cd frontend && npm run lint && npm run build
```

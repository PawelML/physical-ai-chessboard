# One-Ply Reranker / Blunder Classifier Status

Date: 2026-06-15
Status: Phase A implementation ready; ceiling benchmark not run yet

## Implemented Repo Changes

Added Phase A Stockfish-veto infrastructure:

- `arena_core/reranker.py`
  - `RerankedLLMMoveSource` wraps an inner LLM move source.
  - Samples `n_candidates` strict-v7 responses without changing the prompt.
  - Parses with `arena_core.parser.parse_uci_json`.
  - Validates legal moves against `python-chess` `board.legal_moves`.
  - Deduplicates distinct legal UCI moves and tracks multiplicity.
  - Scores candidates through a pluggable scorer.
  - Includes `StockfishVetoScorer`, backed by `StockfishEvaluator.evaluate_move`.
  - Selects the highest-multiplicity non-vetoed candidate; ties use lowest CPL.
  - Falls back to lowest-CPL candidate when all legal candidates are vetoed.
  - Passes through the first raw response when no legal candidate exists.
- `MoveProposal.metadata`
  - Carries reranker telemetry from move source to persistence.
- `attempts.reranker_metadata`
  - New nullable JSON column with Alembic revision
    `0007_attempt_reranker_metadata`.
- Resolver/config wiring
  - New opt-in source prefix: `reranked:<model>`.
  - New env settings:
    - `ARENA_RERANKER_N_CANDIDATES`
    - `ARENA_RERANKER_TEMPERATURE`
    - `ARENA_RERANKER_VETO_CPL_THRESHOLD`
    - `ARENA_RERANKER_VETO_NODES`
    - `ARENA_RERANKER_SCORER`
  - Current scorer implementation: `stockfish`.
  - Tournament config hash and model snapshot sampler params include reranker
    options so reranked runs do not aggregate silently with single-move runs.
- CLI tournament
  - Added `--game-count` so the 100-game ceiling benchmark can be run from CLI.

Per-attempt telemetry now includes:

- `n_candidates_generated`
- `n_legal`
- `n_distinct_legal`
- `n_vetoed`
- `veto_changed_move`
- `all_vetoed`
- `chosen_multiplicity`
- `chosen_uci`
- `first_legal_uci`
- candidate CPL/classification/veto details

## Validation

Passed:

```bash
.venv/bin/ruff check .
.venv/bin/mypy arena_core backend tests
.venv/bin/pytest -q
rm -f /tmp/reranker_migration.db && \
  .venv/bin/alembic -x db_url=sqlite+aiosqlite:////tmp/reranker_migration.db upgrade head
```

Results:

- `ruff`: all checks passed.
- `mypy`: no issues in 41 source files.
- `pytest`: 67 passed, 1 existing Starlette/httpx deprecation warning.
- Alembic: upgraded through `0007_attempt_reranker_metadata`.

## Next Step

Run the Phase A ceiling benchmark against Stockfish 1320 with the current best
GRPO Ollama model:

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
ARENA_STOCKFISH_NODES=200000 \
ARENA_STOCKFISH_HASH_MB=128 \
ARENA_STOCKFISH_SKILL=2 \
ARENA_STOCKFISH_LIMIT_STRENGTH=true \
ARENA_STOCKFISH_TARGET_ELO=1320 \
ARENA_RERANKER_N_CANDIDATES=5 \
ARENA_RERANKER_TEMPERATURE=0.8 \
ARENA_RERANKER_VETO_CPL_THRESHOLD=300 \
ARENA_RERANKER_VETO_NODES=50000 \
arena tournament \
  reranked:chess-ft-qwen35-9b-pilot-grpo-q8_0:latest \
  stockfish \
  --db-url sqlite+aiosqlite:///./arena.db \
  --name qwen35-grpo-reranked-stockfish-veto-100g \
  --legality-mode constrained \
  --game-count 100 \
  --seed 0 \
  --stockfish-skill 2 \
  --stockfish-elo 1320
```

Gate:

- proceed to Phase B only if blunders/game drops sharply and the run scores at
  least one win or draw;
- stop before classifier training if the Stockfish ceiling still scores 0 W/D;
- raise `k` or temperature once if the telemetry shows too few safe candidates.

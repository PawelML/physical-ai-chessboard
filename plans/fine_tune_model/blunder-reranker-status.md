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

## Phase A Ceiling Benchmark

Run completed:

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

Run details:

- Run ID: `18`
- Games: `100`
- Summary rebuild: `Rebuilt 4 game summary rows`
- Wall-clock: approximately 2.5-3 hours; the CLI has no per-game progress log.

Model-perspective result versus the GRPO single-move baseline:

| Metric | Run 17 GRPO single move | Run 18 reranked + Stockfish veto |
| --- | ---: | ---: |
| W-D-L | 0-8-92 | 0-5-95 |
| Avg game plies | 47.26 | 63.69 |
| Model moves/game | 23.42 | 31.62 |
| Avg CPL | 115.00 | 109.22 |
| Blunders/game | 2.03 | 4.25 |
| Blunder rate / model move | 8.67% | 13.44% |
| Illegal attempts | 12 | 3 |
| Malformed attempts | 0 | 0 |

Reranker telemetry for run 18:

| Metric | Value |
| --- | ---: |
| Reranked model attempts | 3165 |
| Avg legal candidates sampled | 4.979 / 5 |
| Avg distinct legal candidates | 2.815 |
| Avg vetoed distinct candidates | 0.780 |
| Veto changed selected move | 1345 / 3165 (42.5%) |
| All legal candidates vetoed | 316 / 3165 (10.0%) |
| Avg chosen multiplicity | 2.753 |

Gate decision:

- **Phase A failed.**
- The reranker did not reduce blunders/game; it increased both blunders/game and
  blunder rate per model move.
- W-D-L also regressed from `0-8-92` to `0-5-95`.
- Avg CPL improved slightly, but this is not enough: the target was a sharp
  blunder drop and at least one win/draw improvement.
- **Do not build Phase B learned classifier from this configuration.**

Interpretation:

- Candidate generation is legal and diverse enough (`~5/5` legal samples,
  `2.8` distinct legal moves), so the failure is not JSON/legality.
- The low-node 1-ply Stockfish veto changes many moves, but the resulting moves
  are not safer under the arena's full 200k-node evaluation.
- Before revisiting Phase B, the selector itself needs a follow-up experiment:
  likely compare `veto_nodes=200000`, select lowest CPL among all safe moves
  instead of highest multiplicity, or run a small offline replay over saved
  candidates to identify why veto-changed moves became blunders.

# Code Review — Simplification & Dead Code

_Date: 2026-06-15 · Scope: `arena_core/`, `backend/`, `frontend/src/`, `finetune/` · Focus: simplification, dead-code removal, de-duplication. **Not** a security or correctness audit._

This is an internal app, so the lens here is **delete what isn't used** and **stop repeating yourself** — not hardening. Findings are grouped by tier (delete → de-duplicate → simplify → leave alone). Every dead-code claim was grep-verified across the whole repo; the six highest-value ones were re-verified by hand this pass and are marked **✓verified**.

Intentional benchmark design (dual UCI/SAN storage, persisted failure attempts, `model_snapshot` keys, flag-gated API providers, documented Phase-0 finetune entry points) is **not** flagged as dead — see Tier 4.

---

## Tier 1 — Delete outright (dead code, near-zero risk)

High-confidence, verified-unused. These are pure deletions with no behavior change.

- [x] **`MoveProposal.source` field is never read.** ✓verified — `rg "\.source\b"` across `arena_core backend tests` returns zero reads (only `source_type`/`source_name`/`move_source`). Every move source sets `source=...` but nothing consumes it.
  - Remove the field at `arena_core/move_sources.py:7` and the `source=` kwargs at `engine.py:50,72,99,339` (+ the Stockfish/LLM source constructions).
- [x] **`session_scope` is dead.** ✓verified — `rg "session_scope"` finds only its definition. The codebase uses `session_factory()` + `async with session.begin()` inline everywhere.
  - Delete `arena_core/persistence/database.py:28-33` and the now-unused `from collections.abc import AsyncIterator` import (line 1).
- [x] **Stockfish `__enter__`/`__exit__` are dead.** ✓verified — no `with StockfishEvaluator`/`with StockfishMoveSource` anywhere (the `with Stockfish...` hits are `StockfishRewardScorer`, an unrelated finetune class). Lifecycle is handled by `.close()` + `__del__`.
  - Delete the four methods at `arena_core/evaluators/stockfish.py:42-47` and `136-141`.
- [x] **`LeaderboardFilters.mode` is dead plumbing.** ✓verified — `fetchLeaderboard` reads `filters.mode` and sets a `mode` query param (`frontend/src/api.ts:406-407`), but the only caller (`App.tsx:85-89`) never sets it.
  - Remove `mode?: string` at `api.ts:128` and the `if (filters.mode)` block at `406-408`. (Leave the unrelated `mode: string` at line 91 — that's a different interface.)
- [x] **Redundant inline import.** ✓verified — `from fastapi import HTTPException` is re-imported inside `get_game` at `backend/main.py:1044`, though it's already imported module-level (line 13).
  - Delete the inline import.
- [x] **`GET /human-games/{id}` (`get_human_game`) has no caller.** `backend/main.py:699` — frontend uses only `GET /human-games` (list) and gets fresh state from move/cancel POST responses; no test hits the bare GET either.
  - Delete the endpoint. _(Confidence: High — but confirm no external client depends on it.)_

**Frontend-orphaned, confirm-then-delete:**

- [x] **`GET /runs` + `RunListItem` and `GET /runs/{run_id}/games`** (`backend/main.py:826,849`, model at `294-301`) are used only by `tests/test_backend.py`, never by the frontend (`api.ts` calls `/runs/compare` and `/runs/{id}/events` only). Either drop them (and `RunListItem`) plus their tests, or keep as an intentional API surface. _(Confidence: High that the UI doesn't use them.)_

---

## Tier 2 — De-duplicate (factor shared helpers)

Repeated logic that has drifted or will drift. Ordered by payoff.

### Cross-module (`arena_core`)

- [x] **`_close_if_present` defined 3× identically** — ✓verified at `engine.py:571`, `tournaments.py:553`, `backend/main.py:1295` (same `getattr(x,"close",None)` + callable check). Define once (e.g. in `move_sources.py`) and import.
- [x] **`template_hash` == `text_hash`** — byte-identical `sha256(x.encode()).hexdigest()` at `prompts.py:42` and `persistence/repositories.py:33`. Collapse to one shared helper.
- [x] **Move-history rendering triplicated** — `_own_moves_for_side`/`_last_opponent_move(_for_side)` exist in `engine.py:503-515`, `finetune/build_dataset.py:371-395`, and `finetune/build_smoke_dataset.py:168-192` (identical). This is the load-bearing prompt-"memory" seam — promote one public helper and have all three call it.

### `backend/main.py` (biggest single-file wins)

- [x] **`_GameMetrics` base model** — `LeaderboardRow`, `RunComparisonRow`, `ModelComparisonRow` (`main.py:303-426`) repeat a ~22-field metrics block (games/wins/draws/losses, rates + `_ci_low`/`_ci_high`, retries, latency, tokens…). Extract a base; the three become base + a few identifying fields. **~120 lines → one base + 3 small extensions.**
- [x] **`_CommonGameKnobs` base** — `StartGameRequest`/`StartStockfishMatchRequest`/`StartHumanGameRequest` (`main.py:99-143`) share a 9-field Ollama/sampling block, repeated a 4th time on `GameJob` and again when each handler builds `GameDefaults(...)` (`527-532,589-594,615-620`). One base + a `.sampling()` helper collapses all of it.
- [x] **`_game_list_item(row)` helper** — `list_games` and `list_run_games` build `GameListItem` from a `Game` row verbatim (`main.py:807-868`), and `_game_stream_payload` (`1129-1140`) hand-builds the same shape as a dict. One helper kills 3 copies and stops REST/SSE drift.
- [x] **`_run_job` lifecycle wrapper** — `_run_game_job` (`main.py:1407`) and `_run_stockfish_match_job` (`1484`) share the entire cancel/fail/`finally: tasks.pop`/finalize scaffold. Extract a wrapper taking the job-specific body. **~50 lines.**
- [x] **Unify `_comparison_row`/`_model_comparison_row`** (`main.py:1864-1994`) — identical `sum(...)` reductions, `wilson_interval` calls, and weighted-average calls; the model variant only adds snapshot + blunder fields. Factor the shared aggregation into one helper (or the `_GameMetrics` base).

### `frontend/src/App.tsx` (3017 lines — split it)

- [x] **Split into ~4 component files** (no rewrite, just move self-contained clusters):
  - `ModelComparison.tsx` — `ModelComparison` + `MATRIX_METRICS` + meta/config helpers (~430 lines, `App.tsx:2554-2991`).
  - `ChessBoard.tsx` — `ChessBoard` + `PlayerStrip` + `DragState`/`PromotionPrompt` + square/promotion helpers (~370 lines, `1667-2017`).
  - `RuntimePanel.tsx` — `RuntimePanel`/`PlayerRuntimeCard`/`ResourceMeter` + formatters.
  - `StartGamePanel.tsx` — `StartGamePanel` + `ModelInput` + `*JobOptions`/`jobLabel`.
  - Plus a shared `format.ts` (`percent`, `movesLabel`, `rateWithCi`, …). **Cuts the monolith roughly in half.**
- [x] **`StockfishLevelSelect` sub-component** — the level `<select>` (Beginner/Club) is written verbatim twice in `StartGamePanel` (`App.tsx:1188-1198` and `1234-1246`).
- [x] **Merge `postJson`/`putJson`** (`api.ts:356-388`) — byte-identical except the HTTP verb → `sendJson(method, ...)`.
- [x] **`resetToReplay(ply?)` callback** — the `setActiveLiveJobId(null); setActiveHumanGameId(null); setIsPlaying(false); setPlyIndex(...)` gesture is repeated across `PlyControls`/`GameList.onSelect`/`togglePlayback`/`MoveListPanel` (`setActiveLiveJobId(null)` appears 10×). Extract one callback.

### `finetune/` (5 build_* + 3 evaluate_* scripts — heavy copy-paste)

- [x] **Create `finetune/_common.py`** for the leaf utilities copy-pasted across scripts (pure deletions, test-covered):
  - `_rate` — 4 copies (`chess_reward:437`, `evaluate_baseline:196`, `evaluate_cpl:232`, `evaluate_lora:222`).
  - `_is_legal_move` — identical in `evaluate_baseline:186` & `evaluate_lora:212` (+ inline in `chess_reward`).
  - `_read_jsonl` — identical in `build_mixed_tactical_dataset:110` & `build_tactical_gate:122`.
  - Stockfish worker init (`_init_worker` + `_WORKER_EVALUATOR`) — identical in `chess_reward:319` & `distill_dataset:213`.
  - `write_metadata_sidecar(path, config, stats)` — the `mkdir` + `write_text(json.dumps({"config":…,"stats":…}))` block repeated in 7+ `main()`s.
- [x] **Shared evaluator core** for `evaluate_baseline`/`_cpl`/`_lora` — same `*EvalStats` (examples + parse/legal/top1 counters & rates), same dataset-iteration + predictions-JSONL `try/finally` loop, same metrics-write/print scaffold. `evaluate_cpl` already imports tokenizer helpers from `evaluate_lora`, so the seam exists; `evaluate_cpl` just extends stats with CPL fields.
- [x] **Route `distill_dataset.py` through `chess_reward.require_stockfish_path`** (`distill_dataset:89-91` re-derives the path that `train_grpo`/`evaluate_cpl` already import) and the shared worker factory.
- [x] **Kill the args→dataclass→kwargs triplication** — worst in `train_grpo.py:64-103` (~35 fields declared in argparse, copied to `GRPOTrainConfig`, then passed positionally). A `config_from_args(DataclassType, args)` helper removes the field-by-field copies here and in `train_lora`/`evaluate_*`.

---

## Tier 3 — Simplify (local readability, low risk)

- [ ] `arena_core/engine.py:257` — drop the `feedback_for_attempt = feedback` alias; pass `feedback` directly to `feedback_given=` (line 270).
- [ ] `arena_core/engine.py:507-515` — `_last_opponent_move_for_side` has two branches returning the same string; reduces to one parity check (`(side == chess.WHITE) != (len(san_history) % 2 == 1)`).
- [ ] `arena_core/cli.py:200-227` — `commit_after_each_ply` branch builds `ArenaGame(...)` with an identical 6-arg body in both `if`/`else`. Build once, branch only on the session-context + flag.
- [ ] `arena_core/tournaments.py:356-377` — `_accepts_rng` introspects the factory signature only so terse test lambdas (`lambda _name: ...`) work. Standardize `SourceFactory` to `(name, rng)`, update the test lambdas, delete `_accepts_rng`.
- [ ] `arena_core/leaderboards.py:180-187` — `wilson_interval(...)` is called twice per metric to unpack `[0]`/`[1]` (3 redundant computations). `lo, hi = wilson_interval(...)` once. Also the `if game.run_id is None: continue` guard at `:54` is redundant given the WHERE clause.
- [ ] `arena_core/evaluators/stockfish.py:91-96` vs `150-157` — engine-version formatting (`f"{name} ({author})" if … else …`) is duplicated across the two classes. Extract `_format_engine_version(engine)`. (And, optionally, a `_StockfishProcess` base for the verbatim `__init__`/`close`/`__del__`/`_ensure_engine` lifecycle.)
- [ ] `frontend/src/App.tsx:1982-1989` — `promotionSymbol`'s `white`/`black` maps are identical; the `color` arg has no effect. Collapse to one map, drop the arg (and `promotion.color` at the call site, `:1935`).
- [ ] `frontend/src/App.tsx:2314-2333` — `modelRuntimeStats` `reduce` recomputes `retries`/`lastUsage`/`averageLatencyMs` identically every iteration; only `totalTokens`/`invalidAttempts` accumulate. Make it a single pass and set the constants directly.
- [ ] `backend/main.py` — replace the verbose field-by-field ORM→Pydantic copies (`leaderboard` at `985-1027` copies 36 fields; same pattern in the `*_out` builders) with `Model.model_validate(row, from_attributes=True)` where names already match, then patch the few computed fields (e.g. `participant`, `*.isoformat()`).
- [ ] `finetune/distill_dataset.py:116-121` — `stats.to_json()` is called twice in `print(...)`; bind it once.

---

## Tier 4 — Leave as-is (intentional design / verified not dead)

Recorded so a future pass doesn't re-flag them:

- **Anthropic & Gemini providers** (`arena_core/llm/providers.py`) — flag-gated behind `ARENA_API_PROVIDERS_ENABLED`; v1 is local-only. Intentionally retained; exercised by `test_llm_providers.py`. Keep.
- **API-key config fields** (`config.py:32-37`) — forward-looking provider config for the gated path. Keep.
- **`smoke_train.py`, `build_smoke_dataset.py`, `scripts/build_finetune_dataset.py`** — documented Phase-0 manual entry points (README/AGENTS/plans + tests reference them). They're _duplication candidates_ (share core logic with `train_lora`/`build_dataset`), **not dead code** — consolidate the shared internals but keep the entry points.
- **`_PromptTemplate` dual `rendered_parts`/`skeleton_parts`** (`prompts.py:22-39`) — backs `test_prompt_template_hash_tracks_template_not_position` (skeleton hash must be position-independent). Keep; only minor nit is giving the two list args `default_factory=list`.
- **Persisted-but-rarely-read columns** (`Model.is_local`/`modality`/`param_size`, dual `accepted_uci`/`accepted_san`, attempt failure rows) — deliberate benchmark metadata per `CLAUDE.md`. Keep.

---

## Suggested sequencing

1. **Tier 1 deletions** first — independent, verified, instantly shrink the surface (one focused commit, run `pytest -q` + `npm run build`).
2. **`finetune/_common.py`** — biggest bang-for-buck de-dup, fully test-covered.
3. **`backend/main.py` base models** (`_GameMetrics`, `_CommonGameKnobs`, `_game_list_item`) — largest single-file reduction (~250+ lines).
4. **`App.tsx` component split** — mechanical, near-zero risk, halves the monolith; do it as its own PR since the diff is large.
5. **Tier 3** local simplifications — opportunistically, alongside whatever file you're already touching.

After any change to scoring/aggregation, run `arena rebuild-summaries`. Note DB/prompt-version impact in PRs per `CLAUDE.md` (none of the above changes prompt templates, so `prompt_version` stays `strict-v7`).

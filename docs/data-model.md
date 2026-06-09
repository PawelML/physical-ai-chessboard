# Data Model — Software Arena (Phase 1)

> Persistence is **attempt-granular**: that is where the benchmark value lives.
> SQLite now (SQLAlchemy 2.0 async + Alembic), written Postgres-ready.
> Leaderboard rows are keyed by an **immutable model snapshot**, never a display name.

## Entity overview

```text
models ──< model_snapshots ──┐
                             │
prompts                      ├──< run_participants >── benchmark_runs
opening_suites ──< opening_lines                          │
                                                          v
                                                        games ──< moves ──< attempts
                                                          │         │
                                                          │         └──< token_usage (1:1 with attempt)
                                                          │
                                                          └ (moves) ──< engine_evaluations
game_summaries  (materialized aggregates for the leaderboard)
provider_errors (optional)
app_settings    (standalone key-value store for UI preferences)
```

## Tables

### `models`
Logical model identity.
- `id`, `provider` (`local`/`anthropic`/`gemini`/`openai`), `name`, `family`,
  `param_size`, `modality` (`text`/`vision`), `is_local` (bool), `notes`.

### `model_snapshots`  ← leaderboard key
Immutable config fingerprint. **All scores reference this, not `models`.**
- `id`, `model_id` (fk), `ollama_digest` / API model id, `quantization`,
  `context_window`, `sampler_params` (json: temperature, top_p, seed, …),
  `runtime_version` (e.g. Ollama version), `created_at`.

### `prompts`
- `id`, `version`, `template_hash`, `mode` (`strict`/`reasoning`),
  `legality_mode` (`open`/`constrained`), `notes`.

### `opening_suites`
- `id`, `name`, `version`, `notes`.

### `opening_lines`
- `id`, `suite_id` (fk), `eco`, `name`, `start_fen` **or** `move_sequence`,
  `intended_ply_start`, `pairing_id` (groups the both-colors pair).

### `benchmark_runs`
- `id`, `name`, `config_hash`, `git_commit`, `hardware_label`, `seed`,
  `stockfish_version`, `stockfish_options` (json: threads, hash, nodes, skill, …),
  `prompt_id` (fk), `opening_suite_id` (fk), `created_at`.

### `run_participants`
One row per competitor side in a run.
- `id`, `run_id` (fk), `model_snapshot_id` (fk, null for Stockfish opponent),
  `opponent_type` (`model`/`stockfish`/`random`), `stockfish_skill` /
  `uci_limit_strength` / `target_elo` (nullable), `color_policy`.

### `games`
- `id`, `run_id` (fk), `white_participant_id` (fk), `black_participant_id` (fk),
  `opening_line_id` (fk), `result` (`1-0`/`0-1`/`1/2-1/2`/`*`),
  `termination_reason` (`checkmate`/`stalemate`/`forfeit_invalid`/`timeout`/…),
  `final_fen`, `pgn`, `started_at`, `ended_at`.

### `moves`
One row per **accepted** move (ply).
- `id`, `game_id` (fk), `ply`, `color`, `fen_before`, `fen_after`,
  `accepted_uci`, `accepted_san`, `legal_move_count`, `move_source`
  (`llm`/`stockfish`/`random`), `retries_used`, `latency_total_ms`.

### `attempts`  ← one row per LLM call (including failures)
- `id`, `move_id` (fk, nullable until a move is accepted), `game_id` (fk),
  `ply`, `attempt_number`, `prompt_id` (fk), `raw_prompt_hash`, `raw_prompt`
  (text, retention-configurable), `raw_response`, `parsed_move` (raw UCI the
  model emitted), `parse_ok`, `legal_ok`, `error_type`
  (`malformed_json`/`illegal_move`/`null`), `feedback_given` (json), `latency_ms`,
  `thinking` (model reasoning trace, nullable), `thinking_used` (bool — whether
  Ollama actually ran thinking; see move-loop doc for the silent-disable gotcha).

### `token_usage`  (1:1 with attempt)
- `attempt_id` (fk), `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `estimated_context_window`, `estimated_context_remaining`, `truncation_applied`,
  `cost_usd` (nullable; 0/local for v1).

### `engine_evaluations`
- `id`, `move_id` (fk), `engine_name`, `engine_version`, `nodes`,
  `depth_reached`, `eval_before_cp`, `eval_after_cp`, `mate_before`, `mate_after`,
  `best_move_uci`, `centipawn_loss`, `classification`
  (`best`/`good`/`inaccuracy`/`mistake`/`blunder`/`mate_*`).

### `game_summaries`  (materialized; rebuilt from the above)
Powers the leaderboard cheaply.
- `id`, `run_id`, `model_snapshot_id`, `color`, `mode`, `legality_mode`,
  `opening_suite_id`, `games_played`, `wins`/`draws`/`losses`, `avg_cpl`,
  `blunders`, `mistakes`, `inaccuracies`, `illegal_rate`, `malformed_rate`,
  `avg_retries`, `forfeit_invalid_count`, `avg_latency_ms`, `total_tokens`.

### `provider_errors` (optional)
- `id`, `run_id`, `attempt_id`, `provider`, `error_kind`, `message`, `created_at`.

### `app_settings`  (UI-managed preferences)
Key-value store (`key` PK, `value` json) for preferences the UI saves, not tied to
any run. Currently holds key `game_defaults` → default sampling/runtime knobs
(`temperature`, `top_p`, `num_ctx`, `num_predict`) used to pre-fill the Start Game
panel. Served by `GET`/`PUT /settings/game-defaults`. This replaced the old fixed
`strict`/`low_creativity` sampling presets with per-knob control. Added in Alembic
revision `0004`.

## Leaderboard query dimensions

Every leaderboard view must be filterable by **all** of:
model snapshot · benchmark run · strict vs reasoning · open vs constrained ·
opponent format · opponent identity · color · opening suite/version · Stockfish
version+options · sampler/temperature config · prompt version · quantization ·
context window.

Without these dimensions the leaderboard looks clean but is scientifically muddy.
**open and constrained are always reported as separate leaderboards.**

## Notes

- Store **prompt and response hashes** even if full raw text retention is later
  trimmed — reproducibility over storage convenience.
- Token counts are **approximate** across heterogeneous local models; never present
  them as exact ground truth.
- Failed attempts are never discarded; they are core benchmark data.

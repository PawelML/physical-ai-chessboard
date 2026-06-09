# Move Loop & Prompt Contracts — Software Arena (Phase 1)

> `python-chess` is the sole source of truth for game state. The LLM never owns
> or mutates state. Every attempt — including failures — is persisted.

## The move loop

```text
for each turn:
  1. build prompt from DB/game state (NOT from chat history)
  2. call model → raw response (model thinking trace, if any, captured per attempt)
  3. parse JSON → extract move string (UCI only)
  4. parse move with chess.Move.from_uci
  5. validate legality against python-chess
  6. if invalid and retries remain:
        send structured feedback, go to step 2 (attempt_number++)
     if invalid and retries exhausted:
        record termination (forfeit_invalid) — do NOT silently fix
     if valid:
        apply move, persist `moves` row, run Stockfish eval
```

The prompt is **rebuilt from game state every turn**, not accumulated as a chat
transcript. This is what satisfies the "memory" requirement *deterministically*:
the model is explicitly given its own move history rather than relying on the
provider's context handling.

## What the prompt contains (strict mode)

- **The win objective.** The strict prompt states the goal explicitly (win by
  checkmate, else play for a draw; pick the strongest legal move, do not hang
  material, look for checks/captures/threats). Without this, models only emit a
  *legal* move, not a *good* one.
- Side to move.
- **Current FEN** (canonical, authoritative).
- Full move history in SAN (game narrative).
- **The model's own prior moves**, listed explicitly in SAN and UCI (the memory
  requirement, made first-class).
- Last opponent move.
- Optional ASCII board (helps weaker/local models).
- Turn number / benchmark rules / output contract.
- Legal moves in UCI: **see legality modes below.**

### Output contract (strict)

UCI coordinate notation is the only accepted move format. Strict JSON:

```json
{ "move": "e2e4" }
```

The `moves` row always stores **both** `accepted_san` and `accepted_uci`,
but the model must emit UCI.

## Legality modes (separate benchmarks)

- **open** — the model is *not* given the legal move list up front; it proposes
  freely and we validate. Measures chess-state understanding and rule-following.
  Illegal-move rate is a first-class metric here.
- **constrained** — the model receives the legal moves (UCI, sorted) and
  must pick one. Measures strategic move quality with legality factored out.

On **retry after an illegal/malformed move, legal moves are always provided**,
regardless of mode.

## Retry protocol

Bounded (default `max_retries = 3`). Structured feedback on failure:

```json
{
  "error": "illegal_move",
  "attempted_move": "e7e5",
  "reason": "e7e5 is not legal in the current position",
  "legal_moves": ["b1c3", "g1f3", "a2a3", "e2e4", "..."],
  "remaining_retries": 2
}
```

- `error` ∈ {`malformed_json`, `illegal_move`}.
- `legal_moves` in feedback is **UCI** (matching the constrained-mode list).
- Every attempt (success or failure) is written to `attempts` with its raw
  response, parse/legality status, feedback given, latency, token usage, and the
  model's thinking trace (`thinking`) plus whether thinking was actually used
  (`thinking_used` — see thinking note below).
- On exhaustion: record `forfeit_invalid` as the game termination. **Never silently
  substitute a move in a scored run.** (An optional separate "assisted continuation"
  mode may substitute a random legal move, but it is not used for scoring.)

## Context strategy as the game grows

- Always include: current FEN, side to move, the model's own moves, last N full
  moves, and (on retry / constrained) legal moves.
- When context pressure hits a small local window, **summarize earlier play as
  structured facts, not prose** — but never let the summary replace FEN:

```json
{
  "opening": "Queen's Gambit Declined structure",
  "material": "White up one pawn",
  "castling_rights": "Black cannot castle",
  "own_moves": ["d2d4", "c2c4", "g1f3"]
}
```

- `telemetry` emits per attempt: prompt tokens, max context, estimated remaining,
  truncation strategy used, fields dropped/compressed. This transparency is part of
  the benchmark.

## reasoning / persona mode (NOT scored)

- A separate template family where the model explains its move and may adopt a
  persona (aggressive tactical / defensive / positional / risk-taking / technician).
- Commentary is an **annotation over an already-scored strict move** — it never
  enters the scoring path, so reasoning verbosity cannot contaminate the benchmark.
- Avoid long natural-language reasoning in scored runs: it raises cost/latency
  without reliably improving move quality.

## Model thinking (Ollama)

When the *thinking* toggle is on, the Ollama request sets `think` and the model's
reasoning trace is captured into `attempts.thinking`. Two gotchas:

- `think="auto"` only **requests** thinking for models whose name contains `qwen`.
- Ollama **silently disables** thinking for models that don't support it (e.g.
  quantized `-ud` builds return HTTP 400 "does not support thinking" and the client
  retries with `think=false`). `attempts.thinking_used` records whether thinking
  actually happened, so a silently-downgraded run is visible rather than assumed.

Thinking is a separate mechanism from the (unscored) reasoning/persona family below:
native thinking is model-internal and discarded from the move JSON; it does not enter
the scored move string.

## Prompt versioning

Every template change bumps `prompt_version` (currently `strict-v7` in `config.py`)
and `template_hash`, both stored on each attempt. Comparisons across prompt versions
are explicit, never accidental. Version history of note: `strict-v5` added the win
objective; `strict-v6` switched the move contract from UCI to SAN; `strict-v7`
returned to UCI-only to keep a single, simpler model contract.

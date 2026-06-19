# LLM-only Deliberation Agent Plan

Date: 2026-06-19
Status: ready to implement
Executor: Codex / Claude Code / another coding agent

This document is intentionally self-contained. It should be possible to hand it to an implementing agent without the surrounding chat context.

## 0. One-paragraph summary

The current Qwen3.5 9B chess policy improved format, legality, average CPL, and game length, but still loses because it acts like a fast single-shot move generator. The next experiment should test whether the same model can improve when allowed to deliberate: generate a move, criticize it, search for opponent replies in language, and then commit a final legal UCI move. The key constraint is that **the final move must still be chosen by the LLM system, not by Stockfish or any engine**. Stockfish may be used as opponent, offline teacher, or post-hoc evaluator, but it must not influence the model's move during a game.

## 1. Goal

Answer the research question:

> Can a small chess-tuned LLM improve its chess result by thinking longer and self-critiquing, without using an engine at play time?

The target is not to build `LLM + Stockfish`. The target is a new benchmark regime:

```text
single-shot LLM  ->  LLM-only deliberative agent
```

Success means a measurable improvement over the current clean single-move GRPO baseline, especially on the failure metric that matters:

- lower `blunders/game`;
- lower `blunder rate / model move`;
- no meaningful regression in legal/malformed attempts;
- better W-D-L or at least more draws against Stockfish 1320.

## 2. Hard contract: what is and is not allowed

### Allowed at play time

- The LLM model itself, called multiple times if needed.
- `python-chess` for rule mechanics only:
  - FEN parsing;
  - legal move generation;
  - validating UCI moves;
  - applying candidate moves to get `fen_after`;
  - SAN formatting;
  - game-over / draw-rule bookkeeping.
- The existing arena opponent, e.g. Stockfish 1320, when the opponent is Stockfish.
- Passive post-hoc evaluator plumbing, as long as the evaluator output is never included in prompts and never affects the move source. If there is any doubt, run the game without a live evaluator and evaluate after the game.

### Not allowed at play time

- Stockfish, Leela, Syzygy, or any chess engine to score, rank, veto, or select candidate moves for the LLM.
- A hand-written tactical evaluator that computes material balance, attacks, pins, SEE, mate search, or other engine-like features and uses them to pick the move.
- Any hidden oracle that inspects engine CPL/classification before the move is accepted.
- Mixing results with the existing strict single-shot leaderboard row. This is a different regime and must be labeled separately.

### Allowed offline

- Stockfish as a teacher for training labels.
- Stockfish as a judge for post-hoc CPL/blunder metrics.
- Stockfish as the benchmark opponent.
- Stockfish for building/evaluating an optional learned self-critic dataset, provided no Stockfish call is made by the move source during the game.

## 3. Context and diagnosis

The existing pipeline already shows that Qwen responds to training: JSON format and legal move compliance improved substantially, and GRPO reduced average CPL. The remaining problem is the tail: the model still makes roughly two decisive blunders per game in clean move-only play.

This plan tests a different hypothesis:

```text
Current result: Qwen as a greedy/short single-pass policy still blunders.
Unanswered:     Qwen as a deliberative LLM-only system may blunder less.
```

This distinction matters. A benchmark with `thinking=false`, short JSON output, and `temperature=0` measures a fast policy head. It does not measure whether the model can improve by spending more tokens on candidate generation, opponent-reply search, and self-checking.

## 4. Experiment arms

Implement and report the arms separately. Do not aggregate them with single-shot runs.

### Arm A — native thinking baseline

Purpose: determine whether simply enabling the model's native thinking path helps.

No new agent logic is required beyond runtime/export support. Run the same move-only prompt with:

- `ARENA_OLLAMA_THINK=on` or equivalent;
- larger `ARENA_OLLAMA_NUM_PREDICT`, e.g. `512` or `1024`;
- sufficient `ARENA_OLLAMA_NUM_CTX`, e.g. `8192`;
- `legality_mode=constrained`;
- strict final JSON parse still required.

Important implementation note: the current Qwen Ollama export template intentionally emits an empty `<think></think>` block for `enable_thinking=False`. For this experiment, add a separate thinking-compatible Modelfile/template or verify that Ollama's `think` parameter produces a valid `response` plus `thinking` without being blocked by the empty-think template.

Expected result: maybe some improvement, but do not assume this will solve blunders. This arm is a baseline for the stronger agentic variants.

### Arm B — two-pass revise-before-final

Purpose: test the cheapest explicit self-check.

Flow per move:

1. Build the normal strict-v7 prompt from the current board.
2. Ask the model for an initial move.
3. Parse and validate the move using the normal parser and `python-chess`.
4. Build a second prompt that includes:
   - the original position;
   - the model's first chosen move;
   - the resulting FEN after that move, computed by `python-chess`;
   - the opponent's legal replies, optionally truncated if too long;
   - explicit instructions to check for mate threats, hanging pieces, queen/king safety, checks, captures, and forcing replies.
5. The model may keep or replace the move.
6. Return only the final UCI JSON to the existing arena move loop.

This remains LLM-only: the second pass is not scored by Stockfish; it is a self-critique prompt.

### Arm C — candidate + critic + final arbiter

Purpose: test a stronger LLM-only agent.

Flow per move:

1. Candidate generation pass:
   - ask the model for `k` legal candidate moves, e.g. 4 or 5;
   - output structured JSON with `move` and a short idea;
   - validate candidates with `python-chess`;
   - drop malformed/illegal candidates;
   - deduplicate by UCI.

2. Critic pass per candidate:
   - apply the candidate move with `python-chess` to obtain `fen_after`;
   - ask the same model to act as the opponent and find the most dangerous reply;
   - ask it to classify the candidate's risk using language-only reasoning, not an engine.

3. Final arbiter pass:
   - give the model the candidate list and its own critic summaries;
   - ask it to pick the candidate with the lowest tactical risk;
   - require strict final JSON: `{"move":"<uci>"}`.

4. Validate final move with the normal arena path.

This is the main experiment. It measures whether explicit deliberation and self-opposition reduce blunders.

### Arm D — optional trained self-critic

Only start this if Arms B/C show a positive signal or at least useful telemetry.

Train a lightweight critic/verifier from offline labels:

```text
(fen_before, candidate_uci, optional fen_after) -> {"risk":"blunder|mistake|playable|good"}
```

Stockfish may label this dataset offline, but the trained critic replaces Stockfish at play time. The final system remains LLM-only at play time if the critic is a learned model, not an engine.

Do not begin Arm D until inference-time deliberation tells us which failure mode dominates:

- self-check never changes moves;
- self-check changes moves but worsens them;
- candidate set lacks safe moves;
- candidate set has safe moves but arbiter cannot identify them;
- critic often identifies the right concern but final pass ignores it.

## 5. New move source design

Add a new module:

```text
arena_core/deliberation.py
```

Suggested public classes/protocols:

```python
DELIBERATIVE_SOURCE_PREFIX = "deliberative:"

@dataclass(frozen=True)
class DeliberationConfig:
    mode: Literal["native_think", "revise", "candidate_critic"] = "revise"
    n_candidates: int = 5
    candidate_temperature: float = 0.7
    critic_temperature: float = 0.0
    final_temperature: float = 0.0
    max_opponent_replies: int = 40
    include_ascii_board: bool = True
    include_legal_moves: bool = True
    max_analysis_tokens: int = 1024
    max_final_tokens: int = 64

class DeliberativeLLMMoveSource:
    source_type = "llm_deliberative"
```

The move source may wrap an inner `LLMMoveSource` or a `CandidateMoveSource`-style protocol with:

```python
async def propose(*, prompt: str, board: chess.Board) -> MoveProposal
```

For minimal integration, wrapping the existing `LLMMoveSource` is acceptable because it already accepts an arbitrary prompt string.

The final `MoveProposal.raw_response` returned to `ArenaGame` must be strict move JSON:

```json
{"move":"e2e4"}
```

All intermediate prompts/responses should go into `MoveProposal.metadata`, not into `raw_response`.

## 6. Prompt contracts

Keep prompts explicit and JSON-shaped. Do not rely on vague "think harder" instructions alone.

### 6.1 Initial move prompt

Use the existing strict-v7 prompt unchanged for Arm B initial move. For Arm C candidate generation, use a new prompt derived from strict-v7 but ask for multiple candidates.

Candidate generation output shape:

```json
{
  "candidates": [
    {"move":"e2e4","idea":"claim center"},
    {"move":"g1f3","idea":"develop and defend"}
  ]
}
```

Parsing should be robust:

- accept a balanced JSON object if extra text appears;
- ignore candidates with missing or invalid `move`;
- lower-case and strip UCI strings;
- validate against `board.legal_moves`;
- deduplicate by UCI.

### 6.2 Revise prompt

Suggested prompt skeleton:

```text
You are reviewing your own chess move before it is played.

Original position:
Side to move: {side}
FEN before: {fen_before}
SAN history: {san_history}
Legal moves: {legal_moves}

Your first chosen move: {candidate_uci} ({candidate_san})
Position after your move: {fen_after}
Opponent legal replies after your move: {opponent_legal_moves}

Check whether your first move is a blunder.
Focus on forcing replies: checks, captures, threats, mate threats, attacks on the queen/king, hanging pieces, and ignored opponent threats.

You may keep the move or replace it with a different legal move from the original legal move list.
Return strict JSON only:
{"move":"e2e4","changed":false,"risk":"short tactical risk note"}
```

The arena parser only needs `move`, but store `changed` and `risk` in metadata if present.

### 6.3 Candidate critic prompt

Suggested prompt skeleton:

```text
You are the opponent trying to refute a candidate chess move.

Original position:
Side to move before candidate: {side}
FEN before: {fen_before}
Candidate move: {candidate_uci} ({candidate_san})
Position after candidate: {fen_after}
Opponent side to move: {opponent_side}
Opponent legal replies: {opponent_legal_moves}

Find the most dangerous opponent reply. Look first for checks, captures, mate threats, attacks on queen/king, and tactics that win material.
Do not use an engine. Reason from the board and legal moves only.

Return strict JSON only:
{
  "candidate":"{candidate_uci}",
  "best_reply":"<uci or null>",
  "risk":"low|medium|high|unknown",
  "blunder_suspected":true,
  "reason":"short concrete reason"
}
```

Validate `best_reply` with `python-chess` if present, but do not use it to compute an engine score.

### 6.4 Final arbiter prompt

Suggested prompt skeleton:

```text
You must now choose one final chess move.

Original position:
FEN: {fen_before}
Side to move: {side}
Legal moves: {legal_moves}

Candidate analyses:
{candidate_analysis_table}

Choose the move with the lowest tactical risk.
Avoid any move where your own critic found a forcing refutation, mate threat, or clear material loss.
Return strict JSON only:
{"move":"e2e4"}
```

Final move must be one of the original legal moves. If the final arbiter returns an illegal/malformed move, fall back to the normal retry path. Do not invent a fallback move unless it was legally proposed and selected by the LLM.

## 7. Runtime configuration

Add settings in `arena_core/config.py`:

```python
deliberation_mode: str = "revise"  # native_think | revise | candidate_critic
deliberation_n_candidates: int = Field(default=5, ge=1, le=16)
deliberation_candidate_temperature: float = Field(default=0.7, ge=0.0)
deliberation_critic_temperature: float = Field(default=0.0, ge=0.0)
deliberation_final_temperature: float = Field(default=0.0, ge=0.0)
deliberation_max_opponent_replies: int = Field(default=40, ge=0)
deliberation_max_analysis_tokens: int = Field(default=1024, gt=0)
deliberation_max_final_tokens: int = Field(default=64, gt=0)
deliberation_persist_intermediate_prompts: bool = True
```

Environment variable examples:

```bash
ARENA_DELIBERATION_MODE=revise
ARENA_DELIBERATION_N_CANDIDATES=5
ARENA_DELIBERATION_MAX_ANALYSIS_TOKENS=1024
```

### Optional but recommended: per-call generation options

Currently `LLMService.complete()` only accepts `(model, prompt)`, while deliberation benefits from different settings per pass: more tokens for analysis, few tokens for final JSON, nonzero temperature for candidates, zero temperature for critic/final.

Preferred change:

```python
@dataclass(frozen=True)
class GenerationOptions:
    temperature: float | None = None
    top_p: float | None = None
    num_predict: int | None = None
    think: str | None = None

class LLMService(ABC):
    async def complete(
        self,
        *,
        model: str,
        prompt: str,
        options: GenerationOptions | None = None,
    ) -> LLMResponse: ...
```

Implementation notes:

- `OllamaLLMService` should merge per-call options over service defaults.
- Non-Ollama providers can ignore unsupported fields, but must still accept the signature.
- Update all existing call sites and tests.
- If this is too large for the first pass, keep global settings and implement DeliberativeLLMMoveSource without per-call overrides. Add a TODO and keep the interface simple.

## 8. Move-source resolver wiring

Add a source prefix:

```text
deliberative:<model-name>
```

Examples:

```text
deliberative:chess-ft-qwen35-9b-pilot-grpo-q8_0:latest
```

Implementation tasks:

- Add `is_deliberative_source_name()` and `inner_source_name()` equivalents, similar to the reranker prefix utilities.
- Wire the prefix in CLI source resolution and backend move-source resolution.
- Ensure `source_type` becomes `llm_deliberative`, not plain `llm`.
- Ensure tournament config hash / model snapshot sampler params include the deliberation regime:

```json
{
  "regime":"llm_deliberative",
  "mode":"candidate_critic",
  "n_candidates":5,
  "candidate_temperature":0.7,
  "critic_temperature":0.0,
  "final_temperature":0.0,
  "max_analysis_tokens":1024
}
```

Leaderboard rows must not aggregate deliberative runs with single-shot runs.

## 9. Metadata and persistence

Use existing `MoveProposal.metadata` to carry deliberation telemetry from the move source to attempts.

The current DB column is named `attempts.reranker_metadata`. For the first implementation, it is acceptable to reuse it with a clear top-level discriminator:

```json
{
  "regime":"llm_deliberative",
  "mode":"candidate_critic",
  "n_candidates_configured":5,
  "n_candidates_generated":5,
  "n_distinct_legal_candidates":4,
  "initial_move":"c2c4",
  "final_move":"g1f3",
  "changed_move":true,
  "candidate_legal_moves":["c2c4","g1f3","d2d4"],
  "critic_summaries":[...],
  "stage_token_usage":{
    "candidate":123,
    "critics":900,
    "final":80
  },
  "stage_latency_ms":{
    "candidate":1200,
    "critics":5000,
    "final":900
  }
}
```

Optional later cleanup: rename or supersede `reranker_metadata` with a generic `attempt_metadata` column. Do not do this migration unless the first implementation needs it.

Persist enough to answer:

- Did deliberation change the move?
- When it changed the move, did blunder rate improve or worsen?
- Did the critic identify high risk on moves that later scored as blunders?
- How many tokens and how much latency did the improvement cost?

## 10. Benchmark commands

Use the clean move-only GRPO baseline as the comparison point. Keep `legality_mode=constrained` and the same opponent settings as the prior legal-list runs.

### 10.1 Arm A — native thinking

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
ARENA_OLLAMA_THINK=on \
ARENA_OLLAMA_NUM_CTX=8192 \
ARENA_OLLAMA_NUM_PREDICT=512 \
ARENA_OLLAMA_TEMPERATURE=0.0 \
ARENA_STOCKFISH_NODES=200000 \
ARENA_STOCKFISH_HASH_MB=128 \
arena tournament \
  chess-ft-qwen35-9b-pilot-grpo-q8_0:latest \
  stockfish \
  --db-url sqlite+aiosqlite:///./arena.db \
  --name qwen35-grpo-native-thinking-100g \
  --legality-mode constrained \
  --game-count 100 \
  --seed 0 \
  --stockfish-skill 2 \
  --stockfish-elo 1320
```

### 10.2 Arm B — two-pass revise

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
ARENA_DELIBERATION_MODE=revise \
ARENA_DELIBERATION_MAX_ANALYSIS_TOKENS=1024 \
ARENA_DELIBERATION_MAX_FINAL_TOKENS=64 \
ARENA_OLLAMA_NUM_CTX=8192 \
ARENA_STOCKFISH_NODES=200000 \
ARENA_STOCKFISH_HASH_MB=128 \
arena tournament \
  deliberative:chess-ft-qwen35-9b-pilot-grpo-q8_0:latest \
  stockfish \
  --db-url sqlite+aiosqlite:///./arena.db \
  --name qwen35-grpo-deliberative-revise-100g \
  --legality-mode constrained \
  --game-count 100 \
  --seed 0 \
  --stockfish-skill 2 \
  --stockfish-elo 1320
```

### 10.3 Arm C — candidate critic

```bash
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
ARENA_DELIBERATION_MODE=candidate_critic \
ARENA_DELIBERATION_N_CANDIDATES=5 \
ARENA_DELIBERATION_CANDIDATE_TEMPERATURE=0.7 \
ARENA_DELIBERATION_CRITIC_TEMPERATURE=0.0 \
ARENA_DELIBERATION_FINAL_TEMPERATURE=0.0 \
ARENA_DELIBERATION_MAX_ANALYSIS_TOKENS=1024 \
ARENA_DELIBERATION_MAX_FINAL_TOKENS=64 \
ARENA_OLLAMA_NUM_CTX=8192 \
ARENA_STOCKFISH_NODES=200000 \
ARENA_STOCKFISH_HASH_MB=128 \
arena tournament \
  deliberative:chess-ft-qwen35-9b-pilot-grpo-q8_0:latest \
  stockfish \
  --db-url sqlite+aiosqlite:///./arena.db \
  --name qwen35-grpo-deliberative-candidate-critic-100g \
  --legality-mode constrained \
  --game-count 100 \
  --seed 0 \
  --stockfish-skill 2 \
  --stockfish-elo 1320
```

## 11. Offline analysis scripts

Add a script:

```text
finetune/analyze_deliberation_run.py
```

Inputs:

```bash
python -m finetune.analyze_deliberation_run \
  --db arena.db \
  --run-id <run_id> \
  --output outputs/finetune/<name>_deliberation_analysis.json
```

Report:

- attempts with deliberation metadata;
- changed-move rate;
- legal candidate count distribution;
- final move blunder rate when `changed_move=true` vs `false`;
- final move blunder rate by self-reported risk bucket;
- average token usage per accepted move;
- average latency per accepted move;
- examples where self-check changed a non-blunder into a blunder;
- examples where self-check changed a blunder into a non-blunder;
- examples where critic flagged `high` but final arbiter still chose the move.

This script is critical. Arena W-D-L is noisy; the metadata tells whether deliberation is mechanistically helping.

## 12. Success gates

Compare to the clean GRPO legal-list baseline, not to mixed strategic-memory runs.

### Arm A gate

Native thinking is worth keeping if it shows at least one of:

- meaningful blunders/game reduction, e.g. >= 10%;
- improved W-D-L/draws without legal regression;
- lower illegal/forfeit rate with similar blunders.

If it only increases latency and does not move blunders, keep it as a negative result.

### Arm B gate

Two-pass revise passes if:

- blunders/game drops by >= 15-20%;
- `changed_move=true` positions have lower post-hoc blunder rate than unchanged positions;
- legal/malformed failures do not materially regress.

If changed moves are worse than unchanged moves, the self-critic is not reliable enough yet. Do not scale to candidate-critic until the failure is understood.

### Arm C gate

Candidate-critic passes if:

- blunders/game drops materially, target <= ~1.5 as first milestone;
- W-D-L improves, ideally more draws or first win;
- candidate telemetry shows the model often proposes multiple legal alternatives;
- final arbiter does not frequently ignore its own high-risk critic warnings.

If Arm C improves blunders but not W-D-L, report the finding: deliberation improves tactical safety, but conversion/endgame skill remains a separate bottleneck.

## 13. Abort conditions

Abort or stop scaling the current configuration if:

- malformed JSON or illegal attempts more than double versus baseline;
- average accepted-move latency becomes too high for 100-game evaluation without a compensating tactical gain;
- `changed_move=true` has a higher blunder rate than `changed_move=false`;
- candidate generation produces fewer than 2 distinct legal candidates on average;
- final arbiter often chooses moves that its own critic marked as high risk;
- native thinking produces long text but the final parser frequently fails.

## 14. Optional Arm D: learned self-critic plan

Only after Arm B/C telemetry is available.

### Dataset builder

Add:

```text
finetune/build_self_critic_dataset.py
```

Rows should be candidate-level, not just position-level:

```json
{
  "fen_before":"...",
  "candidate_uci":"c2c4",
  "candidate_san":"c4",
  "fen_after":"...",
  "source":"arena_blunder|model_sample|stockfish_good|random_legal",
  "centipawn_loss":420,
  "classification":"blunder",
  "target":"{\"risk\":\"blunder\",\"blunder\":true,\"reason_type\":\"material_or_mate\"}"
}
```

Sources:

- actual model blunders from arena runs;
- sampled candidates from the current best model;
- Stockfish best/good moves as safe positives;
- a small number of random legal moves as obvious negatives/positives depending on label.

Balance:

- do not let the dataset become mostly easy safe moves;
- aim for enough blunders/mistakes that the critic learns the tail;
- keep quiet playable positions so it does not mark everything as dangerous.

### Training target

Prefer compact structured outputs over long explanations:

```json
{"risk":"blunder","blunder":true,"reason_type":"mate_or_material"}
{"risk":"playable","blunder":false,"reason_type":"none"}
```

Avoid training long chain-of-thought. If explanations are needed, keep them short and categorical. The runtime agent needs reliable classification more than verbose analysis.

### Runtime integration

Use the learned critic as one pass inside `DeliberativeLLMMoveSource`, but do not call Stockfish. Compare:

```text
candidate_critic_raw_model
candidate_critic_learned_critic
single_shot_baseline
```

## 15. Implementation checklist

### Core code

- [ ] Add `arena_core/deliberation.py`.
- [ ] Add `DELIBERATIVE_SOURCE_PREFIX` and parser helpers.
- [ ] Implement `DeliberativeLLMMoveSource` with modes `native_think`, `revise`, and `candidate_critic`.
- [ ] Keep final `raw_response` strict `{"move":"..."}`.
- [ ] Store all intermediate data in `MoveProposal.metadata`.
- [ ] Validate every candidate/final move with `python-chess`.
- [ ] Do not use Stockfish or engine evaluation in the deliberative move source.

### Config/wiring

- [ ] Add `ARENA_DELIBERATION_*` settings.
- [ ] Wire `deliberative:<model>` in CLI resolver.
- [ ] Wire `deliberative:<model>` in backend resolver.
- [ ] Include deliberation settings in tournament config hash / model snapshot sampler params.
- [ ] Ensure leaderboard rows separate `llm_deliberative` from `llm` and `llm_reranked`.

### Optional LLM service update

- [ ] Add optional per-call generation options to `LLMService.complete`.
- [ ] Implement options merge in `OllamaLLMService`.
- [ ] Update OpenAI/Anthropic/Gemini adapters if their signatures need to match.
- [ ] Add tests or compatibility stubs.

### Export/runtime

- [ ] Add or document a Qwen thinking-compatible Ollama Modelfile/template.
- [ ] Verify `think=on` with `format=json` returns parseable final JSON.
- [ ] Verify `thinking` is persisted separately and does not replace `response` incorrectly.

### Analysis

- [ ] Add `finetune/analyze_deliberation_run.py`.
- [ ] Report changed-move and risk-bucket post-hoc outcomes.
- [ ] Add a status document after first runs:
  - `plans/fine_tune_model/llm-deliberation-agent-status.md`.

### Tests

- [ ] Candidate JSON parser accepts valid candidate lists.
- [ ] Candidate parser drops malformed/illegal moves.
- [ ] Dedup preserves first occurrence and counts multiplicity if implemented.
- [ ] Revise mode returns final strict move JSON.
- [ ] Candidate-critic mode does not call any Stockfish scorer.
- [ ] No-legal-candidate fallback returns the first raw response or fails through normal retry; it must not invent a move.
- [ ] Metadata contains `regime="llm_deliberative"`.
- [ ] Resolver handles `deliberative:<model>`.
- [ ] Config hash changes when deliberation settings change.

Run at least:

```bash
.venv/bin/ruff check .
.venv/bin/mypy arena_core backend tests
.venv/bin/pytest -q
```

## 16. Reporting format

Add a status table like:

| Regime | W-D-L | Avg plies | Avg CPL | Blunders/game | Blunder rate/model move | Illegal attempts | Changed move rate | Tokens/move | Latency/move |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO single-shot legal-list | 0-8-92 | 47.26 | 114.97 | 2.03 | 8.67% | 16 | — | baseline | baseline |
| Native thinking | TBD | TBD | TBD | TBD | TBD | TBD | — | TBD | TBD |
| Deliberative revise | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Candidate critic | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Also include qualitative examples:

- self-check prevented a blunder;
- self-check introduced a blunder;
- critic found a real threat;
- critic hallucinated a threat;
- final arbiter ignored useful criticism.

## 17. Expected outcomes

Possible outcomes and interpretations:

1. **Native thinking helps.**
   The previous benchmark was too short/greedy; keep thinking as a cheap upgrade and test stronger agent modes.

2. **Native thinking does not help, but revise/candidate-critic helps.**
   The model needs structured deliberation, not just more hidden tokens.

3. **Critic identifies risks but final arbiter still chooses bad moves.**
   Train or prompt the final arbiter separately; add stronger rule that high-risk candidates require explicit override.

4. **Candidate set rarely contains safe moves.**
   The policy prior is still too weak; consider candidate generation training or more candidates before critic training.

5. **Deliberation worsens play.**
   The model's verbal analysis is not grounded enough. Treat this as a useful negative result and consider learned critic training from offline labels.

## 18. Why this is a distinct benchmark, not a cheat

This regime is not `LLM + engine`. The model is allowed more inference-time computation, but the chess judgment still comes from the LLM. This is analogous to giving the model time to think, self-review, and compare alternatives. It should be reported honestly as:

```text
LLM-only deliberative agent
```

not as:

```text
strict single-shot model
```

Both numbers are valuable:

- `single-shot` measures raw policy quality;
- `deliberative` measures whether the same LLM can improve by structured self-analysis.

The project should keep both, side by side.
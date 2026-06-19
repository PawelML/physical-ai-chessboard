# Learned Critic/Ranker Training Plan

Date: 2026-06-19
Status: implementation started
Target model family: Qwen3.5 9B chess-tuned policy, plus optional smaller critic

This plan defines the next training experiment after the LLM-only deliberation work. It is
intended to be self-contained for a coding agent.

## 0. Summary

The current best Qwen policy can emit legal moves and has improved average CPL, but it still
loses because the tail contains decisive blunders. Pure prompting helped mechanically only after
fixes: the corrected `candidate_critic` smoke run 25 produced real candidate lists, 0 illegal
attempts, and a 35.5% changed-move rate, but it did not beat the single-shot baseline yet. The next
experiment should train a learned critic/ranker:

```text
(position, legal candidate move) -> calibrated tactical risk / score
```

Stockfish is used offline to create labels. At play time, the arena may call only the LLM policy,
the learned critic model, and `python-chess` for legal move mechanics. No Stockfish call is allowed
inside the move source during games.

Current implementation status:

- candidate-level ranker dataset builder exists and can cache offline Stockfish evaluations;
- position-level choice dataset builder exists for direct "choose the best candidate" SFT;
- first large 5k-position choice experiment shows a real offline signal:
  - generator-first mean CPL: 244.85;
  - 400-step LoRA selected mean CPL: 168.76;
  - 800-step LoRA selected mean CPL: 159.69;
  - oracle target mean CPL: 21.65.

This is not yet strong enough to wire into runtime, but it is better than prompt-only ranking and
is worth one more dataset/model-quality iteration before arena games.

## 1. Research Question

Can a learned critic trained from offline Stockfish labels reduce Qwen's blunder tail when used to
rank Qwen-generated legal candidates at inference time?

Primary target:

- reduce blunders/game and blunder rate/model move versus the clean GRPO single-shot baseline.

Secondary targets:

- preserve legal/malformed attempt rate;
- improve W-D-L or at least increase draws against Stockfish 1320;
- keep accepted-move latency manageable enough for 100-game evaluation.

## 2. Why This Is Different From Previous Attempts

Single-move SFT and GRPO train the policy to emit one move. They improve average quality but still
leave the model without a reliable "is this move tactically unsafe?" mechanism.

Prompt-only deliberation showed two important facts:

- A clean candidate prompt can now generate real candidate sets.
- The model's language-only critic is not reliable enough to rank those candidates consistently.

The learned critic changes the problem from "generate the best move directly" to "recognize risky
candidates." That is closer to how the model is failing: it often has plausible alternatives, but it
does not reliably reject the move that drops material, misses mate, or walks into a forcing reply.

## 3. Hard Contract

Allowed offline:

- Stockfish labels for candidate moves;
- engine CPL, best move, classification, and optional shallow/deeper eval deltas;
- dataset mining from arena DB, PGN corpora, generated candidates, and random legal moves.

Allowed at play time:

- Qwen policy candidate generation;
- learned critic/ranker model inference;
- `python-chess` for FEN parsing, legal move generation, SAN/UCI conversion, and applying moves;
- deterministic selection from learned critic outputs.

Not allowed at play time:

- Stockfish, Leela, Syzygy, or any engine call;
- hand-written tactical scoring such as material balance, attack maps, SEE, mate search, pin
  detection, or custom chess heuristics;
- mixing learned-critic results into existing single-shot leaderboard rows.

The benchmark regime must be labeled separately, e.g. `llm_learned_critic` or
`llm_deliberative_learned_critic`.

## 4. Current Baselines To Beat

Use the same opponent and legality settings as recent runs:

- model: `chess-ft-qwen35-9b-pilot-grpo-q8_0:latest`
- opponent: Stockfish 1320, skill 2
- legality mode: `constrained`
- seed: 0 for smoke comparisons

Known local reference runs:

| Regime | Run | Games | W-D-L | Avg CPL | Blunders/game | Blunder rate | Illegal/malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO single-shot | 19 | 20 | 0-0-20 | 111.85 | 1.75 | 7.51% | 5 / 0 |
| Candidate+critic, initial broken prompt | 22 | 20 | 0-1-19 | 115.96 | 1.80 | 8.29% | 0 / 0 |
| Candidate+critic fixed smoke | 25 | 5 | 0-0-5 | 122.12 | 1.80 | 8.41% | 0 / 0 |

Run 25 is not a quality win, but it is the first useful mechanical baseline:

- average distinct legal candidates: 2.74;
- changed move rate: 35.51%;
- final high-risk chosen count: 1;
- latency: about 5.1s/model move;
- tokens/model move: about 2625.

## 5. Target Architecture

### 5.1 Candidate Generator

Use the current fixed `candidate_critic` candidate prompt:

```text
FEN + side to move + SAN/UCIs + legal moves -> {"candidates":[...]}
```

Do not include the strict single-move prompt inside the candidate prompt.

Runtime settings:

- `n_candidates=5` initially;
- candidate temperature `0.7`;
- final/critic temperature `0.0`;
- fallback to single-shot only when candidate generation yields zero legal candidates.

### 5.2 Learned Critic

The critic receives one candidate at a time:

```text
Input:
  FEN before
  side to move
  candidate UCI and SAN
  FEN after candidate
  legal opponent replies after candidate
  optional candidate idea from generator

Output:
  {"risk":"blunder|mistake|playable|good","blunder":true|false,"score":0-100,"reason_type":"..."}
```

The `score` is a learned risk/quality score, not an engine value at play time. The score is trained
from offline Stockfish labels and used only after model inference.

### 5.3 Selector

Deterministically select the candidate with the best learned critic output:

1. Lowest risk bucket: `good`, `playable`, `mistake`, `blunder`, `unknown`.
2. Highest numeric `score` if available.
3. Tie-break by generator order.

This selector uses no chess heuristics. It only sorts model outputs.

Optional later: use a final arbiter LLM prompt after critic summaries. Do not start there; recent
smoke runs show final arbiter errors are a real failure mode.

## 6. Dataset Design

Add:

```text
finetune/build_critic_ranker_dataset.py
```

Each JSONL row is candidate-level:

```json
{
  "fen_before": "...",
  "side_to_move": "white",
  "candidate_uci": "c2c4",
  "candidate_san": "c4",
  "fen_after": "...",
  "source": "arena_blunder|arena_candidate|policy_sample|stockfish_good|random_legal|opening_suite",
  "legal_move_count": 34,
  "candidate_rank_in_generator": 2,
  "generator_idea": "claim center",
  "stockfish_best_uci": "g1f3",
  "centipawn_loss": 420,
  "classification": "blunder",
  "risk": "blunder",
  "score": 0,
  "target": "{\"risk\":\"blunder\",\"blunder\":true,\"score\":0,\"reason_type\":\"material_or_mate\"}"
}
```

### 6.1 Sources

Use a mix. Do not build only from easy Stockfish best moves.

1. Arena accepted moves from runs 12-25:
   - actual blunders and mate misses;
   - actual good/best moves;
   - positions where candidate+critic changed the move;
   - positions where candidate+critic still blundered.

2. Candidate metadata from deliberative runs:
   - `attempts.reranker_metadata.candidate_legal_moves`;
   - all candidate moves, not only final accepted moves;
   - generator order and candidate idea when available.

3. Fresh policy samples:
   - run the current Qwen candidate generator over selected FENs;
   - temperature 0.7;
   - collect up to 5 legal candidates per position.

4. Stockfish positives:
   - best move;
   - near-best moves where CPL <= 30 or <= 50;
   - include these to prevent the critic from marking every candidate as risky.

5. Random legal negatives/controls:
   - sample 1-3 random legal moves per position;
   - keep labels from Stockfish, because random legal can sometimes be playable.

6. Tactical/error-mined positions:
   - reuse `finetune/build_error_mined_dataset.py` output positions;
   - include known blunder-heavy FENs from GRPO status work.

### 6.2 Position Sampling

Start small but balanced:

```text
Phase 1 smoke:       5k positions, 20k-30k candidate rows
Phase 2 pilot:      25k positions, 100k-150k candidate rows
Phase 3 full:      100k positions, 400k-600k candidate rows
```

Keep splits by position, not by candidate row:

- train: 80%;
- validation: 10%;
- test: 10%.

Do not let candidates from the same FEN appear in both train and eval.

### 6.3 Label Buckets

Use Stockfish `evaluate_move()` and existing arena classifications where possible. Normalize labels:

```text
good:
  classification in {"best", "good"} or CPL <= 50

playable:
  50 < CPL <= 150 or classification == "inaccuracy"

mistake:
  150 < CPL <= 300 or classification == "mistake"

blunder:
  CPL > 300 or classification in {"blunder", "mate_missed"}
```

Mate cases:

- `mate_missed` -> `blunder`;
- if mate classification ambiguity appears, store raw mate fields and audit separately;
- do not train long free-form explanations from ambiguous mate labels.

### 6.4 Reason Types

Keep reason labels categorical and cheap:

```text
none
material_or_mate
king_safety
queen_or_rook_hanging
missed_forcing_reply
endgame_or_pawn_loss
unknown
```

Initial implementation may set `reason_type` mechanically from classification/CPL:

- `mate_missed` -> `material_or_mate`;
- CPL > 300 -> `material_or_mate`;
- otherwise `unknown` or `none`.

Do not attempt hand-written tactical feature extraction in the builder. The point is to train the
model, not to smuggle in an evaluator.

## 7. Prompt/Target Format For Training

Use compact SFT. Do not train long chain-of-thought.

Prompt:

```text
You are a chess move risk classifier.

Position:
FEN before: {fen_before}
Side to move: {side}
Candidate move: {candidate_uci} ({candidate_san})
FEN after candidate: {fen_after}
Opponent legal replies: {opponent_legal_replies}

Classify whether the candidate is tactically safe.
Return JSON only:
{"risk":"blunder|mistake|playable|good","blunder":true,"score":0,"reason_type":"material_or_mate"}
```

Target:

```json
{"risk":"blunder","blunder":true,"score":0,"reason_type":"material_or_mate"}
{"risk":"playable","blunder":false,"score":55,"reason_type":"unknown"}
{"risk":"good","blunder":false,"score":90,"reason_type":"none"}
```

Score mapping:

```text
good:      85-100
playable: 55-84
mistake:  25-54
blunder:   0-24
```

For deterministic labels:

```text
score = clamp(100 - CPL / 4, 0, 100)
```

For mate/missing mate labels, force `score <= 10`.

## 8. Model Choices

### Option A — Same Qwen3.5 9B As Critic

Pros:

- highest likely chess understanding;
- already available locally;
- no extra model family to export.

Cons:

- slow at runtime: one critic call per candidate;
- 5 candidates means 5 additional 9B calls per move.

Use this for the first proof-of-quality pilot.

### Option B — Small Critic Model

Candidate local model: `qwen2.5:1.5b`, or a small Qwen-compatible HF checkpoint if training
environment supports it.

Pros:

- much faster runtime;
- can run as dedicated critic while 9B policy generates candidates.

Cons:

- may not learn enough tactics;
- export/integration adds complexity.

Use only after Option A proves the learned critic signal is useful.

## 9. Training Plan

Add:

```text
finetune/train_critic_ranker.py
finetune/evaluate_critic_ranker.py
```

Training setup:

- QLoRA SFT, same training environment as `finetune/train_lora.py`;
- completion-only loss;
- max sequence length 2048 or 4096 depending on legal reply list size;
- LoRA rank 16 initially;
- learning rate around `2e-4` for smoke, tune after validation;
- 1 epoch for smoke/pilot, avoid overfitting easy labels.

Smoke command sketch:

```bash
python -m finetune.build_critic_ranker_dataset \
  --db arena.db \
  --output data/finetune/critic_ranker_smoke.jsonl \
  --metadata-output data/finetune/critic_ranker_smoke.meta.json \
  --max-positions 5000 \
  --max-candidates-per-position 6 \
  --stockfish-path vendor/stockfish/root/usr/games/stockfish \
  --stockfish-nodes 200000 \
  --stockfish-threads 4 \
  --evaluation-cache data/finetune/critic_ranker_smoke.eval_cache.jsonl \
  --shuffle-positions

python -m finetune.train_critic_ranker \
  --dataset data/finetune/critic_ranker_smoke.jsonl \
  --base-model <hf-qwen35-9b-or-merged-grpo-base> \
  --output-dir outputs/finetune/critic_ranker_smoke \
  --max-steps 200
```

### 9.1 Completed Large Choice Smoke

Built a larger mixed candidate dataset from runs 15-21 and 23-25:

```text
data/finetune/critic_ranker_large_mixed_5k.jsonl
```

Builder settings:

- 5,000 positions from 21,012 seen positions;
- 26,344 candidate rows;
- max 8 candidates per position;
- 4 random legal controls per position;
- Stockfish nodes: 25,000;
- Stockfish threads: 4;
- persistent evaluation cache enabled;
- deterministic shuffled position order with seed 7.

Dataset composition:

| Field | Counts |
| --- | --- |
| risk | blunder 7,647; good 6,135; mistake 4,201; playable 8,361 |
| source | arena_blunder 723; arena_candidate 87; arena_move 4,396; random_legal 17,662; stockfish_good 3,476 |
| split | train 21,012; validation 2,716; test 2,616 |

Converted to position-level choice rows:

```text
data/finetune/critic_choice_large_mixed_5k.jsonl
```

Choice dataset:

- 4,007 positions;
- train 3,188;
- validation 408;
- test 411;
- target risk: good 3,254; playable 750; mistake 3.

### 9.2 Completed Large Choice LoRA Results

Both runs used:

- base model: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`;
- max sequence length: 1536;
- batch size: 2;
- gradient accumulation: 4;
- learning rate: `2e-4`;
- validation examples: 408;
- target format: strict `{"move":"<uci>"}` from the choice dataset.

| Adapter | Steps | Epochs | Parse | Legal | Top-1 oracle | Selected mean CPL | Selected blunders |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `critic_choice_qwen25_15b_large_mixed_lora` | 400 | 1.00 | 100.0% | 99.75% | 37.25% | 168.76 | 102 |
| `critic_choice_qwen25_15b_large_mixed_2epoch_lora` | 800 | 2.01 | 100.0% | 100.0% | 38.24% | 159.69 | 95 |

Reference points on the same 408 validation positions:

- first prompt candidate mean CPL: 244.85;
- oracle target mean CPL: 21.65;
- first prompt candidate blunders: 151;
- 800-step selected blunders: 95.

Interpretation:

- the learned selector is clearly better than generator order offline;
- the second epoch helps modestly instead of collapsing;
- the gap to oracle is still large, so runtime integration should wait until selected mean CPL is
  closer to 100 and selected blunders are below about 70/408 on this validation style.

## 10. Offline Evaluation Gates

Before arena runtime integration, evaluate held-out candidate rows:

Classification metrics:

- blunder recall;
- blunder precision;
- high-risk false negative rate;
- bucket accuracy;
- calibration by score bucket.

Ranking metrics by FEN:

- whether the critic ranks a non-blunder above a blunder when both exist;
- top-1 candidate CPL;
- average selected candidate CPL;
- selected candidate blunder rate.

Minimum smoke gates:

```text
blunder recall >= 70%
blunder precision >= 40%
selected candidate blunder rate < generator-first candidate blunder rate
selected candidate average CPL < generator-first candidate average CPL
```

Do not run a long arena benchmark until offline ranking beats generator order.

## 11. Runtime Integration

Add a new deliberation mode:

```text
ARENA_DELIBERATION_MODE=learned_critic
```

or a separate source prefix:

```text
learnedcritic:<policy-model>
```

Recommended first implementation:

- extend `DeliberativeLLMMoveSource` with mode `learned_critic`;
- keep the same candidate generator;
- call a critic `LLMService`/model per legal candidate;
- parse strict critic JSON;
- select deterministically from critic outputs;
- final `raw_response` remains `{"move":"<uci>"}`.

New settings:

```python
critic_model: str | None = None
critic_temperature: float = 0.0
critic_max_tokens: int = 64
critic_min_safe_score: int = 55
critic_unknown_policy: str = "allow_last"  # or "penalize"
```

Metadata:

```json
{
  "regime": "llm_learned_critic",
  "policy_model": "...",
  "critic_model": "...",
  "candidate_legal_moves": ["..."],
  "critic_outputs": [
    {"candidate":"e2e4","risk":"good","score":92},
    {"candidate":"b1c3","risk":"blunder","score":8}
  ],
  "selected_move": "e2e4",
  "critic_changed_move": true,
  "fallback_reason": null
}
```

## 12. Arena Evaluation Plan

### 12.1 Mechanical Smoke

Run 5 games:

```bash
ARENA_DELIBERATION_MODE=learned_critic \
ARENA_CRITIC_MODEL=<critic-model> \
arena tournament \
  deliberative:chess-ft-qwen35-9b-pilot-grpo-q8_0:latest \
  stockfish \
  --db-url sqlite+aiosqlite:///./arena.db \
  --name qwen35-grpo-learned-critic-smoke-5g \
  --legality-mode constrained \
  --game-count 5 \
  --seed 0 \
  --stockfish-skill 2 \
  --stockfish-elo 1320
```

Gate:

- 0 malformed attempts;
- 0 illegal attempts from final source;
- average distinct legal candidates >= 2.5;
- critic changes moves in at least 10% of candidate-rich positions;
- no frequent final selection of critic-labeled `blunder`.

### 12.2 20-Game Pilot

Compare to run 19 and corrected run 25:

- same seed;
- same Stockfish settings;
- same model.

Pass if:

```text
blunders/game < 1.75
blunder rate < 7.51%
no legal/malformed regression
avg CPL not worse by more than 5%
```

### 12.3 100-Game Benchmark

Only after 20-game pass:

- 100 games;
- compare W-D-L, blunders/game, blunder rate, avg CPL, latency, tokens/move;
- write `plans/fine_tune_model/learned-critic-ranker-status.md`.

## 13. Analysis Script

Add:

```text
finetune/analyze_critic_ranker_run.py
```

Report:

- candidate count distribution;
- selected risk bucket distribution;
- blunder rate by critic risk bucket;
- changed vs unchanged post-hoc blunder rate;
- examples:
  - critic prevented a blunder;
  - critic selected a blunder;
  - critic marked final move safe but Stockfish says blunder;
  - no-legal-candidate fallback;
  - final fallback cases.

This analysis is critical because W-D-L is noisy at 20 games.

## 14. Implementation Checklist

Dataset:

- [x] Add `finetune/build_critic_ranker_dataset.py`.
- [x] Reuse `StockfishEvaluator` / `StockfishRewardScorer` and `(fen,uci)` cache.
- [x] Pull candidate rows from `attempts.reranker_metadata`.
- [ ] Add fresh policy candidate sampling mode.
- [x] Split by FEN, not candidate row.
- [x] Emit JSONL plus metadata summary.
- [x] Add position-level `finetune/build_critic_choice_dataset.py`.

Training:

- [ ] Add `finetune/train_critic_ranker.py`.
- [x] Add `finetune/evaluate_critic_ranker_lora.py`.
- [x] Completion-only loss on compact JSON target for position-level choice SFT.
- [ ] Add tests for prompt/target rendering.

Runtime:

- [ ] Add `learned_critic` deliberation mode.
- [ ] Add critic model settings.
- [ ] Parse critic JSON robustly.
- [ ] Deterministic selection from learned outputs only.
- [ ] Store critic telemetry in `MoveProposal.metadata`.
- [ ] Ensure no Stockfish import/call in runtime move source path.

Benchmarks:

- [ ] 5-game smoke.
- [ ] 20-game pilot.
- [ ] 100-game benchmark only if gates pass.
- [ ] Status document with metrics and examples.

## 15. Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Critic memorizes easy Stockfish best moves but fails on model candidates | Balance dataset around actual Qwen candidates and arena blunders. |
| Critic labels too many moves as dangerous | Include quiet playable/good positions and score calibration. |
| Runtime too slow | Start with 9B critic for signal, then distill to smaller critic if signal is positive. |
| Candidate set lacks safe moves | Increase `n_candidates`, train candidate generator, or add policy sampling diversity. |
| Critic output malformed | Strict JSON target, short completion, fallback to generator order with metadata. |
| Learned critic still picks blunders | Analyze false negatives, mine those positions into the next dataset round. |

## 16. Recommended Next Action

The initial offline ranking proof is positive but not strong enough for arena runtime. The next
action should improve data quality before writing `learned_critic` runtime code:

1. Mine the 800-step false positives/false negatives from validation predictions.
2. Add fresh Qwen policy candidate sampling so the dataset contains more realistic Qwen alternatives
   and fewer purely random legal controls.
3. Build a 25k-position pilot dataset with the Stockfish evaluation cache.
4. Train either:
   - the same Qwen2.5 1.5B choice selector for a faster data-quality check, or
   - a Qwen3.5 9B LoRA critic/selector if the 25k dataset improves validation balance.
5. Only wire runtime `learned_critic` after offline selected mean CPL is near 100 and selected
   blunders are materially below the current 95/408.

The most important early signal is not Elo. It is:

```text
For positions with at least one safe and one unsafe Qwen-generated candidate,
does the critic rank the safe candidate above the unsafe one?
```

If that answer is yes, the approach has a realistic path to reducing blunders/game.

## 17. Implementation Progress

2026-06-19:

- Added `finetune/build_critic_ranker_dataset.py`.
- Added `tests/test_build_critic_ranker_dataset.py`.
- Built a local ignored smoke dataset from arena run `25`:
  - command uses `--max-positions 20`, `--max-candidates-per-position 6`,
    `--random-legal-per-position 1`, and `--stockfish-nodes 50000`;
  - output: `data/finetune/critic_ranker_smoke.jsonl`;
  - metadata: `data/finetune/critic_ranker_smoke.meta.json`;
  - result: 66 candidate rows from 20 positions.
- Smoke row distribution:
  - risk: 33 good, 18 playable, 6 mistake, 9 blunder;
  - source: 20 arena_move, 19 arena_candidate, 11 stockfish_good,
    15 random_legal, 1 arena_blunder.
- Added `finetune/analyze_critic_ranker_dataset.py`.
- Built a larger local ignored pilot dataset from fixed candidate-critic runs `24`
  and `25`:
  - command uses `--max-positions 200`, `--max-candidates-per-position 6`,
    `--random-legal-per-position 1`, and `--stockfish-nodes 50000`;
  - output: `data/finetune/critic_ranker_pilot_fixed_200.jsonl`;
  - metadata: `data/finetune/critic_ranker_pilot_fixed_200.meta.json`;
  - analysis: `data/finetune/critic_ranker_pilot_fixed_200.analysis.json`;
  - result: 652 candidate rows from 200 positions.
- Pilot signal:
  - 56/200 positions include Qwen-generated candidate alternatives;
  - 113/200 positions include both safe and unsafe candidates;
  - final arena moves: mean CPL 94.09, 31 blunders;
  - oracle candidate from each candidate set: mean CPL 7.41, 4 blunders;
  - oracle vs final move: mean CPL reduction 87.37, improved by CPL in 113/200
    positions;
  - oracle vs first generator candidate: mean CPL reduction 102.88 over 56
    positions with Qwen candidates.
- Added `finetune/evaluate_critic_ranker_lora.py`.
- Added `finetune/analyze_critic_choice_predictions.py`.
- Ran two local ignored Qwen2.5 1.5B LoRA critic probes:
  - unbalanced train split:
    `outputs/finetune/critic_ranker_qwen25_15b_pilot_lora`;
  - class-balanced oversampled train split:
    `outputs/finetune/critic_ranker_qwen25_15b_balanced_lora`.
- Held-out validation result on 64 rows:
  - unbalanced: 64/64 JSON parse, risk match 43.75%, blunder match 85.94%,
    mean score absolute error 28.06;
  - unbalanced ranking: selected mean CPL 116.5 vs oracle 4.5,
    oracle move match 4/20 positions;
  - unbalanced failure mode: all 64 predictions were `risk=good`.
  - balanced: 64/64 JSON parse, risk match 25.0%, blunder match 73.44%,
    mean score absolute error 30.64;
  - balanced ranking: selected mean CPL 91.39 vs oracle 4.5,
    oracle move match 7/20 positions;
  - balanced failure mode: predictions became diverse, but overcalled blunders
    and still ranked candidates poorly.
- Built a larger local ignored candidate-level dataset from runs `23`, `24`,
  and `25`:
  - output: `data/finetune/critic_ranker_candidate_runs_23_25.jsonl`;
  - metadata: `data/finetune/critic_ranker_candidate_runs_23_25.meta.json`;
  - result: 2519 candidate rows from 595 positions.
- Added the simpler position-level choice task:
  - builder: `finetune/build_critic_choice_dataset.py`;
  - output: `data/finetune/critic_choice_candidate_runs_23_25.jsonl`;
  - metadata: `data/finetune/critic_choice_candidate_runs_23_25.meta.json`;
  - result: 419 choice rows, 337 train, 42 validation, 40 test;
  - target prompt index is now distributed across positions 1-8. An earlier
    local run sorted candidates by quality and leaked the answer; that adapter
    is ignored and should not be used.
- Ran a local ignored Qwen2.5 1.5B LoRA choice probe:
  - adapter: `outputs/finetune/critic_choice_qwen25_15b_shuffled_lora`;
  - validation predictions:
    `outputs/finetune/critic_choice_qwen25_15b_shuffled_predictions.jsonl`;
  - analysis:
    `outputs/finetune/critic_choice_qwen25_15b_shuffled_analysis.json`.
- Held-out choice validation result on 42 rows:
  - JSON parse: 42/42;
  - legal moves: 42/42;
  - candidate-list moves: 42/42;
  - oracle top-1 match: 19/42, 45.24%;
  - selected mean CPL: 200.74;
  - first prompt candidate mean CPL: 205.13;
  - oracle target mean CPL: 7.95.

Conclusion after the choice probe: the model learned JSON format and candidate
membership, but not chess ranking. It improved only marginally over choosing the
first displayed candidate and remains far from oracle. Do not wire
`learned_critic` yet.

Next implementation step: generate substantially more policy candidates, not
just reuse the small candidate-critic runs. The next dataset should target at
least 20k-30k candidate rows / several thousand choice positions before another
training run.

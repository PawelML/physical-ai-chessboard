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
- first large 5k-position choice experiment showed a real offline signal:
  - generator-first mean CPL: 244.85;
  - 400-step LoRA selected mean CPL: 168.76;
  - 800-step LoRA selected mean CPL: 159.69;
  - oracle target mean CPL: 21.65.
- the current strongest offline selector is the pairwise-composed 1,200-step adapter on the
  policy1k validation split:
  - pairwise-composed selected mean CPL: 116.64;
  - top-1 oracle: 170/408 (41.67%);
  - selected blunders: 72/408;
  - selected high-risk moves: 117/408.
  - hard-case regression: 81/300 top-1, 224.93 selected mean CPL, 89/300 selected blunders.

This is a meaningful offline improvement over prompt-only ranking and direct list-choice SFT. It is
still not ready to trust in arena runtime until latency/throughput is fixed and a small arena smoke
confirms the offline signal transfers to full games.

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

### 9.3 Completed 800-Step Hard-Case Mining

Extended `finetune/analyze_critic_choice_predictions.py` to emit aggregate error diagnostics and a
JSONL file of the worst validation cases:

```text
outputs/finetune/critic_choice_qwen25_15b_large_mixed_2epoch_hard_cases.jsonl
```

Hard-case command:

```bash
python -m finetune.analyze_critic_choice_predictions \
  --dataset data/finetune/critic_choice_large_mixed_5k.validation.jsonl \
  --predictions outputs/finetune/critic_choice_qwen25_15b_large_mixed_2epoch_predictions.jsonl \
  --output outputs/finetune/critic_choice_qwen25_15b_large_mixed_2epoch_analysis.json \
  --hard-cases-output outputs/finetune/critic_choice_qwen25_15b_large_mixed_2epoch_hard_cases.jsonl \
  --hard-cases-limit 100 \
  --hard-regret-threshold 200
```

Additional validation diagnostics:

- improved vs first candidate: 200/408;
- worsened vs first candidate: 99/408;
- tied first candidate: 47/408;
- mean CPL regret vs oracle: 139.38;
- selected high-risk moves: 156/408;
- selected blunders: 95/408;
- missed-good cases: 126/408;
- high-regret cases (`selected_cpl - oracle_cpl >= 200`): 93/408.

Risk transitions show the main failure mode:

| Oracle risk -> selected risk | Count |
| --- | ---: |
| good -> good | 163 |
| good -> playable | 50 |
| good -> mistake | 54 |
| good -> blunder | 72 |
| playable -> playable | 39 |
| playable -> mistake | 6 |
| playable -> blunder | 23 |
| mistake -> mistake | 1 |

The top hard cases are not parser or legality failures. They are ranker preference failures where a
good move is already visible in the candidate list, but the model selects a random-looking blunder.
The next dataset iteration should oversample these exact structures:

- one or more high-CPL random/legal decoys;
- at least one good or playable candidate in the same list;
- target good move sometimes appears at prompt index 1, sometimes later;
- include repeated hard-case FENs with shuffled candidate order to reduce position/order shortcuts.

### 9.4 Completed Hard-Case Regression Eval Set

Added `finetune/build_critic_choice_hardcase_eval.py` to turn mined hard cases into a normal
candidate-choice eval dataset. This is deliberately an eval/regression set, not a training
augmentation, because the source hard cases came from validation predictions.

Generated local regression set:

```text
data/finetune/critic_choice_large_mixed_5k_hardcase_eval.jsonl
```

Builder command:

```bash
python -m finetune.build_critic_choice_hardcase_eval \
  --input outputs/finetune/critic_choice_qwen25_15b_large_mixed_2epoch_hard_cases.jsonl \
  --output data/finetune/critic_choice_large_mixed_5k_hardcase_eval.jsonl \
  --metadata-output data/finetune/critic_choice_large_mixed_5k_hardcase_eval.meta.json \
  --variants-per-case 3 \
  --seed 11
```

Regression set composition:

- input hard cases: 100;
- output rows: 300;
- target risk: good 96 hard cases, playable 4 hard cases;
- category counts across source hard cases: high_regret 93, missed_good 96, selected_blunder 63.

Current 800-step adapter baseline on this regression set:

| Metric | Value |
| --- | ---: |
| Parse rate | 100.0% |
| Legal rate | 100.0% |
| Top-1 oracle | 13/300 (4.33%) |
| Selected mean CPL | 376.93 |
| Oracle mean CPL | 14.31 |
| Mean CPL regret vs oracle | 362.62 |
| Selected high-risk moves | 267/300 |
| Selected blunders | 166/300 |
| Missed-good cases | 255/300 |

This is now the regression gate for the next data/model iteration. A candidate adapter is not a
real improvement unless it improves both:

- the normal validation set (`critic_choice_large_mixed_5k.validation.jsonl`);
- the hard-case regression set above.

Minimum next-adapter hard-case target:

- top-1 oracle above 20%;
- selected mean CPL below 250;
- selected blunders below 100/300;
- no parse/legal regression.

### 9.5 Completed Fresh Qwen Policy Candidate Sampling Smoke

Added:

```text
finetune/sample_policy_candidates.py
```

and extended:

```text
finetune/build_critic_ranker_dataset.py --candidate-input <jsonl>
```

This allows fresh Qwen-generated candidate lists to be sampled offline, then labeled through the
same Stockfish/cache pipeline as arena and random candidates.

Smoke sample command:

```bash
python -m finetune.sample_policy_candidates \
  --input data/finetune/critic_choice_large_mixed_5k.train.jsonl \
  --output data/finetune/policy_candidates_qwen35_grpo_train100.jsonl \
  --metadata-output data/finetune/policy_candidates_qwen35_grpo_train100.meta.json \
  --model chess-ft-qwen35-9b-pilot-grpo-q8_0:latest \
  --split train \
  --max-positions 100 \
  --samples-per-position 1 \
  --n-candidates 5 \
  --temperature 0.7 \
  --num-predict 512 \
  --timeout-seconds 180
```

Sampling result:

- positions sampled: 100/100;
- requests: 100;
- service errors: 0;
- legal candidate rows: 312;
- mean candidates/request: 3.12;
- candidate count distribution: 1:36, 2:9, 3:6, 4:5, 5:44.

Label command:

```bash
python -m finetune.build_critic_ranker_dataset \
  --db arena.db \
  --output data/finetune/critic_ranker_policy_qwen35_grpo_train100.jsonl \
  --metadata-output data/finetune/critic_ranker_policy_qwen35_grpo_train100.meta.json \
  --stockfish-path vendor/stockfish/root/usr/games/stockfish \
  --stockfish-nodes 25000 \
  --stockfish-threads 4 \
  --evaluation-cache data/finetune/critic_ranker_large_mixed_5k.eval_cache.jsonl \
  --run-id -1 \
  --candidate-input data/finetune/policy_candidates_qwen35_grpo_train100.jsonl \
  --max-positions 100 \
  --max-candidates-per-position 8 \
  --random-legal-per-position 0 \
  --no-stockfish-best \
  --shuffle-positions \
  --seed 11
```

Labeled policy-sample result:

- rows: 312;
- positions: 100;
- risk counts: blunder 100, mistake 59, playable 105, good 48;
- mixed safe/unsafe positions: 41;
- first Qwen candidate mean CPL: 205.17;
- oracle within Qwen candidate list mean CPL: 136.62;
- oracle gain vs first Qwen candidate: 66.95 CPL over 100 positions;
- first Qwen candidate was a blunder while candidate-list oracle was safe: 8 positions.

Interpretation:

- Fresh policy sampling works and produces legal candidates without arena games.
- Qwen's own candidate list often contains a better move than its first candidate, so learned
  ranking has real room to help.
- This smoke set is train split only; it can seed the next training dataset without contaminating
  validation or hard-case regression.

### 9.6 Completed 1k Fresh Policy Candidate Training Probe

Scaled fresh policy sampling from 100 to 1,000 train positions using:

```text
chess-ft-qwen35-9b-pilot-grpo-q8_0:latest
```

Sampling result:

- requests: 1,000;
- positions with at least one legal candidate: 996;
- service errors: 0;
- legal candidate rows: 2,790;
- candidate count distribution: 0:4, 1:406, 2:90, 3:117, 4:62, 5:321.

Labeled policy-sample result:

- rows: 2,790 from 996 positions;
- risk counts: blunder 929, mistake 599, playable 850, good 412;
- first Qwen candidate mean CPL: 247.06;
- oracle within Qwen candidate list mean CPL: 178.33;
- oracle gain vs first Qwen candidate: 68.9 CPL over 996 positions;
- first Qwen candidate was a blunder while candidate-list oracle was safe: 87 positions.

Combined this with the previous large mixed dataset:

```text
data/finetune/critic_ranker_large_mixed_plus_policy1k.jsonl
data/finetune/critic_choice_large_mixed_plus_policy1k.jsonl
```

The combined choice dataset still has 4,007 positions, but candidate lists for train positions are
enriched with Qwen-generated alternatives:

- train 3,188;
- validation 408;
- test 411;
- target risk: good 3,260; playable 744; mistake 3.

Trained an 800-step Qwen2.5 1.5B choice LoRA:

```text
outputs/finetune/critic_choice_qwen25_15b_large_mixed_policy1k_800_lora
```

Normal validation result on the 408-position validation set:

| Metric | Previous 800-step | Policy1k 800-step |
| --- | ---: | ---: |
| Parse rate | 100.0% | 100.0% |
| Legal rate | 100.0% | 99.75% |
| Top-1 oracle | 156/408 (38.24%) | 154/408 (37.75%) |
| Selected mean CPL | 159.69 | 170.93 |
| Oracle mean CPL | 21.65 | 21.65 |
| Mean CPL regret vs oracle | 139.38 | 150.49 |
| Selected high-risk moves | 156/408 | 160/408 |
| Selected blunders | 95/408 | 101/408 |
| Missed-good cases | 126/408 | 130/408 |
| High-regret cases | 93/408 | 96/408 |

Hard-case regression result:

| Metric | Previous 800-step | Policy1k 800-step |
| --- | ---: | ---: |
| Parse rate | 100.0% | 100.0% |
| Legal rate | 100.0% | 100.0% |
| Top-1 oracle | 13/300 (4.33%) | 24/300 (8.0%) |
| Selected mean CPL | 376.93 | 352.44 |
| Oracle mean CPL | 14.31 | 14.31 |
| Mean CPL regret vs oracle | 362.62 | 338.13 |
| Selected high-risk moves | 267/300 | 254/300 |
| Selected blunders | 166/300 | 159/300 |
| Missed-good cases | 255/300 | 243/300 |
| High-regret cases | 255/300 | 236/300 |

Interpretation:

- Fresh policy candidates helped the hard-case regression set, but only modestly.
- The same adapter regressed on the normal validation set.
- The hard-case result is still far below the minimum next-adapter gate of top-1 above 20%,
  selected mean CPL below 250, and selected blunders below 100/300.
- Do not wire this adapter into runtime. Treat it as evidence that realistic Qwen candidates are
  useful, but 1k sampled train positions are not enough and the current target format is probably
  too weak.

### 9.7 Completed Pairwise Contrastive Choice Probe

Added:

```text
finetune/build_critic_pairwise_dataset.py
```

This builder converts candidate-level rows into two-candidate contrastive examples. Each example
contains one safe candidate (`good` or `playable`) and one unsafe candidate (`mistake` or `blunder`)
with a configurable CPL gap. The target remains strict:

```json
{"move":"<uci>"}
```

so the existing LoRA trainer and evaluator can be reused.

Dataset command:

```bash
python -m finetune.build_critic_pairwise_dataset \
  --input data/finetune/critic_ranker_large_mixed_plus_policy1k.jsonl \
  --output data/finetune/critic_pairwise_large_mixed_plus_policy1k.jsonl \
  --metadata-output data/finetune/critic_pairwise_large_mixed_plus_policy1k.meta.json \
  --pairs-per-position 4 \
  --min-cpl-gap 100
```

Dataset result:

- rows: 15,067 pairwise examples;
- positions with pairs: 3,999/5,000;
- split: train 12,009; validation 1,523; test 1,535;
- pair risks: blunder->good 5,899; blunder->playable 4,781; good->mistake 3,189;
  mistake->playable 1,198;
- target prompt index: position 1 = 7,548; position 2 = 7,519.

Training command:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 python -m finetune.smoke_train \
  --dataset data/finetune/critic_pairwise_large_mixed_plus_policy1k.train.jsonl \
  --output-dir outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_400_lora \
  --max-steps 400 \
  --max-seq-length 1024 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4
```

Training result:

- base model: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`;
- steps: 400;
- epoch fraction: 0.2664;
- train loss: 0.06552.

Also trained a longer 1,200-step run on the same pairwise train split:

```text
outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora
```

Longer training result:

- steps: 1,200;
- epoch fraction: 0.7993;
- train loss: 0.05627.

Full validation result on 1,523 held-out pairwise examples:

| Metric | 400-step | 1,200-step |
| --- | ---: | ---: |
| Parse rate | 100.0% | 100.0% |
| Legal rate | 100.0% | 100.0% |
| Candidate-list rate | 100.0% | 100.0% |
| Top-1 target match | 971/1,523 (63.76%) | 1,058/1,523 (69.47%) |
| First-prompt baseline top-1 | 770/1,523 (50.56%) | 770/1,523 (50.56%) |
| Selected mean CPL | 164.20 | 142.41 |
| First-prompt candidate mean CPL | 239.45 | 239.45 |
| Target/oracle mean CPL | 39.27 | 39.27 |
| Mean CPL regret vs oracle | 125.40 | 103.82 |
| Selected high-risk moves | 553/1,523 | 466/1,523 |
| First-prompt high-risk moves | 752/1,523 | 752/1,523 |
| Selected blunders | 386/1,523 | 325/1,523 |
| First-prompt blunders | 538/1,523 | 538/1,523 |
| High-regret cases | 336/1,523 | 278/1,523 |
| Missed-good cases | 284/1,523 | 234/1,523 |

Interpretation:

- Pairwise contrastive training gives a much clearer offline signal than the previous list-ranking
  target.
- The 1,200-step model beats balanced prompt-order baseline by about 18.9 percentage points.
- It reduces mean CPL by about 97 and selected blunders by 213 on the validation pairs.
- Longer training improved every tracked validation metric versus 400 steps, without parse/legal
  regression.
- This is still not runtime-ready: selected mean CPL 142 and 325/1,523 blunders are too high.
- The next experiment should keep the pairwise target, scale realistic Qwen candidate pairs, and
  test pairwise scoring as an intermediate signal for a multi-candidate selector offline.

### 9.8 Completed Pairwise-Composed Multi-Candidate Smoke

Added:

```text
finetune/build_pairwise_comparison_eval.py
finetune/analyze_pairwise_comparison_predictions.py
```

This expands a normal multi-candidate choice row into all unordered candidate pairs, runs the
pairwise adapter on each pair, then composes the final choice by pairwise wins. Tie-break is the
original candidate order. The selector does not use Stockfish labels, CPL, risk buckets, or scores
at selection time.

Smoke expansion command:

```bash
python -m finetune.build_pairwise_comparison_eval \
  --input data/finetune/critic_choice_large_mixed_plus_policy1k.validation.jsonl \
  --output data/finetune/critic_choice_large_mixed_plus_policy1k.validation100_pairwise_eval.jsonl \
  --metadata-output data/finetune/critic_choice_large_mixed_plus_policy1k.validation100_pairwise_eval.meta.json \
  --max-choice-rows 100
```

Expansion result:

- source choice rows: 100;
- pair rows: 1,262;
- source candidate counts: 2:2, 3:1, 4:6, 5:30, 6:60, 7:1;
- pair target prompt index: position 1 = 624, position 2 = 638.

Pairwise prediction used:

```text
outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora
```

Pair-level result on those 1,262 generated pairs:

- parse/legal: 100.0%;
- pair target match: 758/1,262 (60.06%).

Composed selection result on the original 100 multi-candidate rows:

| Metric | First candidate | List-choice 800-step | Pairwise-composed 1,200-step |
| --- | ---: | ---: | ---: |
| Parse/legal | n/a | 100.0% / 99.0% | 100.0% / 100.0% at pair level |
| Candidate-list rate | n/a | 99.0% | 100.0% at pair level |
| Top-1 oracle | n/a | 35/100 (35.0%) | 38/100 (38.0%) |
| Mean CPL | 264.11 | 179.30 | 128.19 |
| Oracle mean CPL | 19.48 | 19.48 | 19.48 |
| Mean CPL regret vs oracle | n/a | 160.50 | 111.24 |
| High-risk selected | 54/100 | 38/100 | 33/100 |
| Blunders selected | 39/100 | 24/100 | 19/100 |
| High-regret cases | n/a | 25/100 | 16/100 |
| Missed-good cases | n/a | 33/100 | 26/100 |

Interpretation:

- Pairwise composition transfers to normal multi-candidate rows better than the direct list-choice
  adapter on this 100-row smoke slice.
- The largest win is not top-1; it is risk reduction: mean CPL 128 vs 179 and blunders 19 vs 24
  against the list-choice adapter.
- Pair-level accuracy drops from 69.47% on the dedicated pairwise validation set to 60.06% on pairs
  generated from normal choice rows, so distribution mismatch is still real.
- This smoke is too small to justify runtime integration, but it is strong enough to run the full
  408-row validation and hard-case regression composition next.

### 9.9 Completed Pairwise-Composed Full Normal Validation

Full expansion command:

```bash
python -m finetune.build_pairwise_comparison_eval \
  --input data/finetune/critic_choice_large_mixed_plus_policy1k.validation.jsonl \
  --output data/finetune/critic_choice_large_mixed_plus_policy1k.validation_pairwise_eval.jsonl \
  --metadata-output data/finetune/critic_choice_large_mixed_plus_policy1k.validation_pairwise_eval.meta.json
```

Expansion result:

- source choice rows: 408/408;
- pair rows: 5,112;
- source candidate counts: 2:5, 3:5, 4:26, 5:144, 6:220, 7:4, 8:4;
- pair target prompt index: position 1 = 2,574, position 2 = 2,538.

Pairwise prediction used:

```text
outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora
```

Pair-level result on 5,112 generated pairs:

- parse/legal/candidate-list: 100.0%;
- pair target match: 3,121/5,112 (61.05%).

Composed selection result on the original 408 multi-candidate validation rows:

| Metric | First candidate | List-choice 800-step | Pairwise-composed 1,200-step |
| --- | ---: | ---: | ---: |
| Parse/legal | n/a | 100.0% / 99.75% | 100.0% / 100.0% at pair level |
| Candidate-list rate | n/a | 99.75% | 100.0% at pair level |
| Top-1 oracle | n/a | 154/408 (37.75%) | 170/408 (41.67%) |
| Mean CPL | 244.85 | 170.93 | 116.64 |
| Oracle mean CPL | 21.65 | 21.65 | 21.65 |
| Mean CPL regret vs oracle | n/a | 149.28 | 96.67 |
| High-risk selected | 222/408 | 160/408 | 117/408 |
| Blunders selected | 151/408 | 101/408 | 72/408 |
| High-regret cases | n/a | unknown | 58/408 |
| Missed-good cases | n/a | unknown | 84/408 |

Interpretation:

- The 100-row smoke result transfers to the full frozen validation set.
- The pairwise-composed selector is materially better than the direct list-choice 800-step adapter
  on all tracked quality metrics: +16 top-1 selections, -54.29 mean CPL, -29 blunders, and
  -43 high-risk selections.
- Pair-level accuracy remains modest at 61.05%, but majority-style composition over all candidate
  pairs produces a stronger final selector than asking the model to choose from the whole list.
- This is the first offline learned-selector result that looks plausibly useful for reducing
  blunders/game. It still needs hard-case regression before runtime integration, because the
  validation set is not enough to prove the tail is controlled.

### 9.10 Completed Pairwise-Composed Hard-Case Regression

Hard-case expansion command:

```bash
python -m finetune.build_pairwise_comparison_eval \
  --input data/finetune/critic_choice_large_mixed_5k_hardcase_eval.jsonl \
  --output data/finetune/critic_choice_large_mixed_5k_hardcase_pairwise_eval.jsonl \
  --metadata-output data/finetune/critic_choice_large_mixed_5k_hardcase_pairwise_eval.meta.json
```

Expansion result:

- source choice rows: 300/300;
- pair rows: 3,909;
- source candidate counts: 4:18, 5:93, 6:183, 7:6;
- pair target prompt index: position 1 = 1,903, position 2 = 2,006.

Pair-level result on 3,909 generated pairs:

- parse/legal/candidate-list: 100.0%;
- pair target match: 2,105/3,909 (53.85%).

Composed selection result on the original 300 hard-case regression rows:

| Metric | First candidate | List-choice 800-step | Pairwise-composed 1,200-step |
| --- | ---: | ---: | ---: |
| Parse/legal | n/a | 100.0% / 100.0% | 100.0% / 100.0% at pair level |
| Candidate-list rate | n/a | 100.0% | 100.0% at pair level |
| Top-1 oracle | n/a | 24/300 (8.0%) | 81/300 (27.0%) |
| Mean CPL | 243.40 | 352.44 | 224.93 |
| Oracle mean CPL | 14.31 | 14.31 | 14.31 |
| Mean CPL regret vs oracle | n/a | 338.13 | 210.62 |
| High-risk selected | 167/300 | 254/300 | 142/300 |
| Blunders selected | 103/300 | 159/300 | 89/300 |
| High-regret cases | n/a | 236/300 | 130/300 |
| Missed-good cases | n/a | 243/300 | 133/300 |

Interpretation:

- The pairwise-composed selector passes the previous hard-case minimum gate: top-1 is above 20%,
  selected mean CPL is below 250, and selected blunders are below 100/300.
- This is a large improvement over the direct list-choice 800-step adapter, especially in the
  failure tail: -127.51 mean CPL, -112 high-risk selections, and -70 blunders.
- It is also better than the first candidate baseline in mean CPL and blunders, although it still
  worsens 103/300 rows versus first prompt order. The selector is useful but not yet uniformly
  reliable.
- Runtime integration should not start as a full benchmark mode yet. The immediate engineering
  need is a cached/batched pairwise scorer so offline iteration and a tiny arena smoke are not
  bottlenecked by one-pair-at-a-time inference.

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
- [x] Add fresh policy candidate sampling mode.
- [x] Split by FEN, not candidate row.
- [x] Emit JSONL plus metadata summary.
- [x] Add position-level `finetune/build_critic_choice_dataset.py`.
- [x] Add pairwise contrastive `finetune/build_critic_pairwise_dataset.py`.

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

The initial offline ranking proof is positive but was not strong enough for arena runtime. The 1k
fresh policy-candidate probe improved the hard-case regression numbers, but regressed the normal
validation set. The pairwise contrastive probe is stronger: the 1,200-step adapter beats balanced
prompt-order baseline by about 18.9 percentage points on dedicated validation pairs, cuts
validation-pair blunders from 538 to 325, and the pairwise-composed selector transfers to the full
408-row normal validation set.

The current best offline selector is:

```text
Qwen2.5 1.5B pairwise critic, 1,200-step LoRA, composed by pairwise wins
```

On the frozen 408-row normal validation set it reaches 170/408 top-1 oracle choices, 116.64
selected mean CPL, 72/408 selected blunders, and 117/408 selected high-risk moves. This beats the
direct list-choice 800-step adapter on the same validation style. On the 300-row hard-case
regression set it reaches 81/300 top-1, 224.93 selected mean CPL, 89/300 selected blunders, and
142/300 selected high-risk moves, beating the direct list-choice adapter by a wide margin.

The next action should be runtime-enabling engineering plus a very small arena smoke, not a full
benchmark mode yet:

1. Add a cached/batched pairwise scorer so composed selection can evaluate all pairs for one
   candidate list without one Python/model call per pair.
2. Add a hidden/experimental arena move-source mode that generates Qwen candidates, runs the
   learned pairwise selector, logs every pair decision, and falls back to generator order on any
   model failure. Keep it separate from existing leaderboard modes.
3. Run only a tiny 5-game Stockfish smoke first, measuring accepted-move latency, candidate count,
   changed-move rate, illegal/malformed count, average CPL, and blunders/game.
4. If the tiny smoke is mechanically clean and not slower than an acceptable threshold, run the
   20-game Stockfish comparison against the GRPO single-shot and previous candidate+critic runs.
5. In parallel, scale fresh Qwen policy sampling to a 10k-25k train-position pilot, reusing the
   Stockfish cache.
6. Build pairwise examples only from positions with at least one safe Qwen candidate and at least
   one unsafe Qwen candidate; single-candidate positions add little ranking signal.
7. Keep the existing validation and hard-case regression sets frozen, and add a frozen pairwise
   validation/test report.
8. Consider a Qwen3.5 9B LoRA critic/selector only after the small selector proves the data format
   scales.

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

2026-06-19 later:

- Sampled 1,000 fresh train positions with the Qwen3.5 GRPO policy candidate
  generator:
  - output: `data/finetune/policy_candidates_qwen35_grpo_train1k.jsonl`;
  - result: 2,790 legal candidates from 996 positions, with zero service
    errors.
- Labeled those candidates through the existing Stockfish/cache pipeline:
  - output: `data/finetune/critic_ranker_policy_qwen35_grpo_train1k.jsonl`;
  - risk counts: 929 blunder, 599 mistake, 850 playable, 412 good.
- Built a combined mixed-plus-policy1k ranker/choice dataset and trained:
  - adapter:
    `outputs/finetune/critic_choice_qwen25_15b_large_mixed_policy1k_800_lora`;
  - training: 800 steps, 2.01 epochs, train loss 0.1502.
- Normal validation result on 408 frozen validation positions:
  - parse 100.0%, legal 99.75%;
  - top-1 oracle 154/408 (37.75%);
  - selected mean CPL 170.93;
  - selected blunders 101/408.
- Hard-case regression result on 300 frozen variants:
  - parse 100.0%, legal 100.0%;
  - top-1 oracle 24/300 (8.0%);
  - selected mean CPL 352.44;
  - selected blunders 159/300.
- Conclusion: policy candidates help the hard-case set slightly but this
  adapter regresses normal validation and fails the hard-case gate. Do not
  integrate it into runtime; scale targeted fresh policy sampling and test a
  stronger pairwise/contrastive target next.

2026-06-19 pairwise iteration:

- Added `finetune/build_critic_pairwise_dataset.py`.
- Added `tests/test_build_critic_pairwise_dataset.py`.
- Built local ignored pairwise dataset:
  - output: `data/finetune/critic_pairwise_large_mixed_plus_policy1k.jsonl`;
  - rows: 15,067;
  - splits: 12,009 train, 1,523 validation, 1,535 test;
  - target prompt index is balanced: 7,548 at position 1 and 7,519 at
    position 2.
- Trained local ignored pairwise Qwen2.5 1.5B adapter:
  - adapter:
    `outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_400_lora`;
  - training: 400 steps, 0.2664 epoch, train loss 0.06552.
- Full pairwise validation result on 1,523 examples:
  - parse 100.0%, legal 100.0%, candidate-list 100.0%;
  - top-1 target match 971/1,523 (63.76%);
  - selected mean CPL 164.20 vs first-prompt 239.45 and oracle 39.27;
  - selected blunders 386/1,523 vs first-prompt 538/1,523.
- Conclusion: pairwise contrastive training is a materially better target than
  list-choice SFT for the small selector. It is not runtime-ready yet, but the
  next training iteration should scale this pairwise format rather than return
  to the previous list-only target.

2026-06-19 pairwise 1,200-step continuation:

- Trained a longer local ignored adapter on the same pairwise train split:
  - adapter:
    `outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora`;
  - training: 1,200 steps, 0.7993 epoch, train loss 0.05627.
- Full pairwise validation result on the same 1,523 examples:
  - parse 100.0%, legal 100.0%, candidate-list 100.0%;
  - top-1 target match 1,058/1,523 (69.47%);
  - selected mean CPL 142.41 vs first-prompt 239.45 and oracle 39.27;
  - selected blunders 325/1,523 vs first-prompt 538/1,523;
  - selected high-risk moves 466/1,523 vs first-prompt 752/1,523.
- Delta vs 400-step pairwise adapter:
  - top-1 +87 examples, +5.71 percentage points;
  - selected mean CPL improved by 21.79;
  - selected blunders down by 61;
  - high-regret cases down by 58.
- Conclusion: longer pairwise training scales positively and is close to the
  70% validation target, but it still needs lower CPL/blunder tail before
  runtime integration. The next concrete step should test pairwise-composed
  ranking on multi-candidate choice rows offline.

2026-06-19 pairwise-composed selector smoke:

- Added `finetune/build_pairwise_comparison_eval.py`.
- Added `finetune/analyze_pairwise_comparison_predictions.py`.
- Added `tests/test_pairwise_comparison_eval.py`.
- Built a local ignored 100-row composed-ranking smoke set:
  - source: `data/finetune/critic_choice_large_mixed_plus_policy1k.validation.jsonl`;
  - output:
    `data/finetune/critic_choice_large_mixed_plus_policy1k.validation100_pairwise_eval.jsonl`;
  - result: 1,262 pair rows from 100 source choice rows.
- Ran pairwise predictions with:
  - adapter:
    `outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora`;
  - output:
    `outputs/finetune/critic_choice_large_mixed_plus_policy1k_validation100_pairwise_composed_1200_pair_predictions.jsonl`.
- Pair-level result:
  - parse 100.0%, legal 100.0%;
  - pair target match 758/1,262 (60.06%).
- Composed selector result on the original 100 choice rows:
  - top-1 oracle 38/100;
  - selected mean CPL 128.19;
  - selected blunders 19/100;
  - selected high-risk moves 33/100.
- Same 100-row direct list-choice 800-step baseline:
  - top-1 oracle 35/100;
  - selected mean CPL 179.30;
  - selected blunders 24/100;
  - selected high-risk moves 38/100.
- Conclusion: pairwise-composed ranking is a better offline selector than the
  direct list-choice adapter on this smoke slice. The next step is the full
  408-row validation plus the 300-row hard-case regression set.

2026-06-19 pairwise-composed full validation:

- Built the full local ignored pairwise expansion:
  - source: `data/finetune/critic_choice_large_mixed_plus_policy1k.validation.jsonl`;
  - output:
    `data/finetune/critic_choice_large_mixed_plus_policy1k.validation_pairwise_eval.jsonl`;
  - result: 5,112 pair rows from 408 source choice rows.
- Ran pairwise predictions with:
  - adapter:
    `outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora`;
  - output:
    `outputs/finetune/critic_choice_large_mixed_plus_policy1k_validation_pairwise_composed_1200_pair_predictions.jsonl`.
- Pair-level result:
  - parse 100.0%, legal 100.0%, candidate-list 100.0%;
  - pair target match 3,121/5,112 (61.05%).
- Composed selector result on the original 408 choice rows:
  - top-1 oracle 170/408 (41.67%);
  - selected mean CPL 116.64;
  - selected blunders 72/408;
  - selected high-risk moves 117/408.
- Full-validation direct list-choice 800-step baseline:
  - top-1 oracle 154/408 (37.75%);
  - selected mean CPL 170.93;
  - selected blunders 101/408;
  - selected high-risk moves 160/408.
- Conclusion: the pairwise-composed selector also wins on the full frozen
  normal validation set, not only the 100-row smoke slice. The next required
  gate is hard-case regression before any runtime arena integration.

2026-06-19 pairwise-composed hard-case regression:

- Built the local ignored hard-case pairwise expansion:
  - source: `data/finetune/critic_choice_large_mixed_5k_hardcase_eval.jsonl`;
  - output: `data/finetune/critic_choice_large_mixed_5k_hardcase_pairwise_eval.jsonl`;
  - result: 3,909 pair rows from 300 source hard-case choice rows.
- Ran pairwise predictions with:
  - adapter:
    `outputs/finetune/critic_pairwise_qwen25_15b_large_mixed_policy1k_1200_lora`;
  - output:
    `outputs/finetune/critic_choice_large_mixed_5k_hardcase_pairwise_composed_1200_pair_predictions.jsonl`.
- Pair-level result:
  - parse 100.0%, legal 100.0%, candidate-list 100.0%;
  - pair target match 2,105/3,909 (53.85%).
- Composed selector result on the original 300 hard-case rows:
  - top-1 oracle 81/300 (27.0%);
  - selected mean CPL 224.93;
  - selected blunders 89/300;
  - selected high-risk moves 142/300.
- Full hard-case direct list-choice 800-step baseline:
  - top-1 oracle 24/300 (8.0%);
  - selected mean CPL 352.44;
  - selected blunders 159/300;
  - selected high-risk moves 254/300.
- Conclusion: the pairwise-composed selector passes the previously defined
  hard-case gate and is the first learned selector worth a small runtime smoke.
  Do not run a full benchmark yet; first add batched/cached scoring and measure
  latency on a tiny arena run.

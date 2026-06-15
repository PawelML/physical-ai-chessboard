# Pilot Analysis (Qwen3.5 9B, 20k) and the Case for Stockfish Distillation

> Decision record written 2026-06-12, after benchmarking the 20k-example pilot
> fine-tune (`chess-ft-qwen35-9b-pilot-q8_0`) against Stockfish 1320. Read this
> before choosing the next training run — it explains *why* the next step is
> engine-distilled labels rather than more human data or a bigger model, and it
> doubles as interview material for explaining the training process.

## 1. Pilot results (20 games each vs Stockfish beginner, from `arena.db`)

| Metric | qwen3.5:9b base | chess-ft pilot (20k) |
|---|---|---|
| W-D-L | 0-2-18 | 0-0-20 |
| forfeit_invalid | 5 | 0 |
| illegal attempts | 18.0% | 2.6% |
| malformed attempts | 6.4% | 0% |
| avg CPL | ~200 | ~124 |
| top-1 accuracy | 29.4% | 44.4% |
| avg plies/game | 28.6 | 41.9 |
| **blunders / game** | **~1.9 (in far shorter games)** | **~2.0** |
| mistakes / game | ~2.1 | ~2.7 |

(The base model's 2 draws at n=20 are noise, not an advantage — it also lost
5 games without making a single scored mistake, by forfeit.)

## 2. Diagnosis

The 20k pilot solved the **cheap half of the problem**: JSON format, UCI
legality, game stability. That half is now near ceiling (0 forfeits, 2.6%
illegal) and will not yield further wins.

What remains is **tactical move quality**: the fine-tune still makes ~2
blunders + ~2.7 mistakes per game (40 blunders / 414 evaluated moves across 20
games). Against a 1320 opponent, roughly one serious blunder per game is
fatal; at two per game the win probability is ~0 regardless of how clean the
rest of the play is. **Blunder rate per game is the single metric the next
run must move.**

## 3. Why the blunders won't go away with more of the same data

The teacher is noisy. Measured on our own pilot positions with Stockfish at
100k nodes (`finetune/distill_dataset.py` label stats):

- 2000-rated Lichess players match Stockfish's best move only **~48–55%** of
  the time;
- their average CPL on our sampled positions is ~30–40, and the tail includes
  outright mistakes and blunders (time trouble, blitz).

Behaviour cloning learns the teacher's *distribution*, including that tail.
Scaling 20k → 200k human examples buys a slow, log-shaped improvement at
~10× the training cost (~65 h on the 3090), while the label noise — the thing
that causes the blunders — stays constant.

## 4. Decision: change the labels, not (yet) the scale or the model

Ranked by expected value per hour:

1. **Stockfish-distilled labels (chosen).** Same positions, same byte-identical
   strict-v7 prompts, but the completion becomes Stockfish's best move at
   fixed nodes. Attacks the measured failure mode (blunders) directly; this is
   the recipe behind DeepMind's "Grandmaster-Level Chess Without Search"
   (270M params → ~2300 Elo). Labeling is **CPU-only and cheap**: measured
   ~25 rows/s with 8 workers at 100k nodes ⇒ the full 190k train set relabels
   in ~2 h, while the 3090 stays free.
2. **Blunder-filtered human labels** (`--label-mode filter`): keeps human
   diversity, drops examples losing > max CPL. Softer variant; useful as an
   ablation ("clean imitation vs distillation").
3. **More human data (200k)** — only if distillation stalls; prefer renting an
   H100 (~$25–40, overnight) over 65 h of 3090 time.
4. **Gemma 4 12B — explicitly deferred.** Switching model now would change two
   variables at once; Qwen already proved it responds to training, the E4B
   Gemma attempt gave no positive signal, and 12B trains slower/tighter on
   24 GB. Revisit as an ablation once the recipe is settled.

## 5. Next-run playbook (commands)

```bash
# 1. Relabel the existing pilot/main datasets (CPU, ~2h for 190k):
export ARENA_STOCKFISH_PATH="$PWD/vendor/stockfish/root/usr/games/stockfish"
python -m finetune.distill_dataset \
  --input data/finetune/lichess_2000_2024-01_main.train.jsonl \
  --output data/finetune/lichess_2000_2024-01_distilled.train.jsonl \
  --metadata-output data/finetune/lichess_2000_2024-01_distilled.train.meta.json \
  --label-mode distill --nodes 100000
# ...and the same for the .val.jsonl file.

# 2. Train on a 50k slice first (~16h on the 3090), same hyperparameters as
#    the pilot — one variable changes: the labels.

# 3. Offline proxy eval before any arena games (fast feedback):
#    legality + top-1 vs engine label on held-out val.

# 4. Arena benchmark: >=50 games per color vs Stockfish beginner.
```

## 6. Methodology guardrails (also the interview answers)

- **One variable per run.** Pilot → distilled run changes labels only: same
  positions, same prompts, same hyperparameters, same model. Any improvement
  is attributable.
- **n=20 games cannot distinguish "never wins" from "wins 5%".** Benchmark
  conclusions need ≥50 games per color; the leaderboard's Wilson intervals
  make this visible (`low_sample` flag).
- **Iterate on offline proxies, not full games.** Held-out legality / top-1 /
  blunder-rate cost minutes; 20 arena games cost hours. Full benchmarks are
  for confirmed candidates.
- **The benchmark contract never moves.** All gains must show up under the
  unchanged strict-v7 prompt and UCI-only parser — otherwise we'd be measuring
  a different benchmark, not a better model.
- **Dedup after distillation.** Distilled labels are deterministic, so
  repeated positions (especially openings) collapse to identical
  (prompt, completion) rows; the relabeling script drops exact duplicates by
  default to avoid silently oversampling openings.

## 7. FAQ: "If it's trained on Stockfish's moves, can it beat Stockfish?"

Likely interview question; the answer has two parts.

**General principle — imitation cannot exceed the teacher.** Behaviour cloning
approximates the teacher's move distribution, and every approximation is
lossy: the student has no search, only patterns, so it lands *below* the
teacher's strength. DeepMind's distilled model reached ~2300 Elo from a ~3500
Elo Stockfish teacher — excellent, yet 1200 Elo below the source. Surpassing a
teacher requires going beyond imitation (RL / self-play, à la AlphaZero).

**Why that's fine here — teacher and opponent are two different Stockfishs.**
The benchmark opponent is Stockfish *deliberately crippled* to 1320 Elo
(Skill Level 2 randomly picks inferior candidate moves — it blunders by
design). The label teacher is full-strength Stockfish at 100k nodes (~3000+
Elo), about 1700 Elo above the opponent. The student can lose 1000+ Elo in
distillation and still land comfortably above 1320. We never need to beat the
teacher — only the much weaker opponent.

**Won't student and opponent play identically?** No: the opponent is
deliberately weakened (different move distribution), the student reproduces
the teacher only partially and adds its own generalization errors, and the
training positions come from human games, so the student's patterns cover
human-typical structures rather than engine search trees.

## 8. Train-from rule for the distilled run

Always train `Qwen base → dataset X`, never continue from a previous LoRA.
Continuing `base → human 20k → distilled` is a two-stage *curriculum* — a
different (third) experiment whose result cannot be attributed to the labels.
Also watch the size confound: comparing 20k human vs 50k distilled changes
labels *and* scale at once. The rigorous arm is `base → 20k distilled` (same
size, same positions, same hyperparameters as the pilot — only the labels
differ); scale up only after that comparison is read. The curriculum variant
(human SFT → distilled SFT) is a legitimate later ablation, kept separate.

## 9. Optional chapter 3: RL with Stockfish as the reward model

Status: idea, not scheduled. Only worth starting after distilled SFT plateaus —
otherwise gains cannot be attributed.

**Full AlphaZero is out of reach and out of contract.** AlphaZero = policy+value
net from scratch + MCTS woven into both play and target generation + millions
of self-play games on thousands of TPUs (community reproduction: Leela Chess
Zero, years of distributed compute). On one 3090 with a 9B LLM it is not
feasible — and the benchmark contract forbids search at play time anyway
(single-move JSON answer).

**The feasible cousin: GRPO with engine reward** (the standard LLM-RL recipe,
with Stockfish substituted for the human/AI reward model):

1. Start from the distilled SFT model (RL needs a policy that already plays
   legally; it will not bootstrap from 18%-illegal output).
2. Per training position, sample k moves (e.g. 8, with temperature).
3. Reward each move with Stockfish: ≈ −CPL, plus a penalty for illegal moves
   and malformed JSON. (`finetune/distill_dataset.py` already implements the
   scoring step in effect.)
4. GRPO update: reinforce moves above the group mean, suppress below, with a
   KL anchor to the SFT model so the JSON format does not degenerate.

Why it fits the hardware: completions are ~10 tokens, so rollouts are cheap;
scoring is CPU; TRL/Unsloth ship GRPO for QLoRA. ~10k positions × 8 samples is
an overnight job on the 3090, bottlenecked by inference, not training.

Why it is interesting beyond the score: **RL has no imitation ceiling** — the
model is rewarded for move quality, not for matching a teacher, so it can in
principle exceed its own SFT stage (never full Stockfish: without search it
stays capacity-bound — the DeepMind lesson). It also targets exactly the
failure metric that matters here, blunders, because they carry the largest
negative reward.

Honest risks: LLM-RL is fragile (format collapse without the KL anchor and
malformed-output penalty); expected gains are "polish", not deep tactics; and
the portfolio claim must stay modest. The upside for the narrative is that the
project would then cover the full modern training chain: **human imitation
(SFT) → distillation from a stronger teacher → RL against an external reward
model** — the same chain frontier models are trained with, on a chessboard and
a single consumer GPU.

## 10. The narrative so far (for the portfolio / interview)

1. Benchmark showed local models lose to a 1320 engine two ways: small models
   by illegal-move forfeits, mid-size by checkmate at ~200 CPL.
2. A 20k LoRA pilot on human games — rendered through the benchmark's own
   prompt builder — eliminated the format/legality failure mode entirely
   (18%→2.6% illegal, 0 forfeits) and cut CPL 200→124. Measured by the same
   harness, same prompt version.
3. The residual failure is ~2 blunders/game, traced to label noise: the human
   teachers themselves agree with the engine only ~50% of the time.
4. Next iteration replaces imitation labels with engine-distilled labels at
   ~zero GPU cost — a hypothesis the harness can confirm or refute in one run.

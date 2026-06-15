# Qwen3.5 9B Stockfish-Distilled Pilot Status

Date: 2026-06-13

## Goal

Train a Qwen3.5 9B LoRA from the base model on the same pilot positions as the
human-label Qwen pilot, but relabel completions with Stockfish best moves. This
keeps the comparison focused on the label source:

- human pilot: same prompts/positions, human moves as targets
- distilled pilot: same prompts/positions, Stockfish moves as targets

## Dataset

Source pilot:

- train: `data/finetune/lichess_2000_2013-12_pilot.train.jsonl`
- val: `data/finetune/lichess_2000_2013-12_pilot.val.jsonl`

Distilled outputs:

- train: `data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl`
- val: `data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl`

Stockfish settings:

- `--label-mode distill`
- `--nodes 100000`
- `--workers 8`
- `--dedup` enabled

Train stats:

- rows read: 18,950
- rows written: 16,696
- duplicates dropped: 2,254
- human/engine agreement: 47.14%
- human average CPL: 38.31

Val stats:

- rows read: 1,050
- rows written: 948
- duplicates dropped: 102
- human/engine agreement: 49.33%
- human average CPL: 40.93

## Training

Output:

- `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora`

Command shape:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 .venv-train/bin/python -m finetune.train_lora \
  --model unsloth/Qwen3.5-9B \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot_distilled.train.jsonl \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora \
  --max-seq-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --eval-steps 0 \
  --save-steps 50 \
  --save-total-limit 2 \
  --logging-steps 5
```

Observed training details:

- examples after filtering/truncation: 16,680
- total steps: 261
- effective batch size: 64
- trainable params: 86,556,672 / 9,496,370,416, about 0.91%
- xFormers active: `Xformers = 0.0.35`, Flash Attention 2 disabled
- pure training runtime: about 5h55m
- end-to-end process time including model download/load: about 6h32m
- final train loss: 0.2512

Final adapter saved:

- `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_lora/adapter_model.safetensors`

## Held-Out 100 Results

All three runs below use the same target labels:

- dataset: `data/finetune/lichess_2000_2013-12_pilot_distilled.val.jsonl`
- limit: 100
- target: Stockfish best move
- `--disable-thinking`
- `--max-new-tokens 24`

| Model | JSON | Legal | Top-1 vs Stockfish |
| --- | ---: | ---: | ---: |
| Base `unsloth/Qwen3.5-9B` | 100% | 48% | 8% |
| Human-label Qwen pilot LoRA | 100% | 92% | 15% |
| Stockfish-distilled Qwen pilot LoRA | 100% | 89% | 16% |

Outputs:

- `outputs/finetune/qwen35_9b_base_on_distilled_pilot_val100_eval.json`
- `outputs/finetune/qwen35_9b_human_lora_on_distilled_pilot_val100_eval.json`
- `outputs/finetune/qwen35_9b_distilled_lora_lichess_2000_pilot_val100_eval.json`

## Interpretation

The distilled pilot clearly beats the base model on the same Stockfish-labeled
held-out set, but it is not yet a decisive improvement over the human-label LoRA
on a 100-position proxy: 16% vs 15% top-1 is effectively a tie at this sample
size. The stronger signal is that both LoRAs fixed most of the base model's move
format/legal-move weakness, while Stockfish imitation needs either more data,
better training settings, or a stronger offline metric such as Stockfish CPL of
generated moves.

Next recommended step: export the distilled adapter to Ollama/GGUF and run arena
games against the existing human-label Qwen Q8 model, then compare blunder rate
and illegal move rate rather than relying only on top-1.

## Ollama Export

Date: 2026-06-13

Final local GGUF files:

- `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_gguf/qwen35_9b_lichess_2000_pilot_distilled.Q8_0.gguf`
- `outputs/finetune/qwen35_9b_lichess_2000_pilot_distilled_gguf/qwen35_9b_lichess_2000_pilot_distilled.Q4_K_M.gguf`

Ollama models:

- `chess-ft-qwen35-9b-pilot-distilled-q8_0:latest`
- `chess-ft-qwen35-9b-pilot-distilled-q4_k_m:latest`

Export notes:

- `finetune.export_ollama` failed during Unsloth GGUF conversion because the
  local Unsloth/llama.cpp converter hit `ModelBase.__init__() got an unexpected
  keyword argument 'target_model_dir'`.
- Workaround: use the merged HF files produced by Unsloth, convert directly to
  `Q8_0` with `convert_hf_to_gguf.py`, then create `Q4_K_M` from Q8 with
  `llama-quantize --allow-requantize`.
- `gemma4:12b-it-bf16` was removed from Ollama to free space for model
  validation/import. The q4/q8 Gemma 12B Ollama variants remain installed.
- Smoke check: `chess-ft-qwen35-9b-pilot-distilled-q4_k_m` returned strict JSON
  for a trivial move prompt.
- Backend `/models` lists both distilled models, so the Vite app should show
  them after refresh.

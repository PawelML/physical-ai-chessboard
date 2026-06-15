# Gemma4 E4B Fine-Tuning Status

Date: 2026-06-12

## Scope

Train and export a current Gemma4 model smaller than the 12B class, using the
same 20k Lichess pilot dataset and strict-v7 prompt contract used for the
Qwen3.5 9B pilot.

## Model

- HF/Unsloth: `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit`
- Local Ollama exports:
  - `chess-ft-gemma4-e4b-pilot-q4_k_m`
  - `chess-ft-gemma4-e4b-pilot-q8_0`
- Chat turn markers used for response-only loss:
  - instruction: `<|turn>user\n`
  - response: `<|turn>model\n`

## Training

Command summary:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
.venv-train/bin/python -m finetune.train_lora \
  --model unsloth/gemma-4-E4B-it-unsloth-bnb-4bit \
  --train-dataset data/finetune/lichess_2000_2013-12_pilot.train.jsonl \
  --output-dir outputs/finetune/gemma4_e4b_lichess_2000_pilot_lora \
  --max-seq-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --eval-steps 0 \
  --save-steps 50 \
  --save-total-limit 2 \
  --logging-steps 5 \
  --instruction-part $'<|turn>user\n' \
  --response-part $'<|turn>model\n'
```

Result:

- examples after truncation/masking: 18,939
- total optimizer steps: 296
- trainable parameters: 73,400,320 / 8,069,556,768 (0.91%)
- runtime: 9287 seconds
- final train loss: 0.0257
- adapter: `outputs/finetune/gemma4_e4b_lichess_2000_pilot_lora`

## Held-Out LoRA Sanity

100 examples from `data/finetune/lichess_2000_2013-12_pilot.val.jsonl`:

| Model path | JSON | Legal | Top-1 |
|---|---:|---:|---:|
| `outputs/finetune/gemma4_e4b_lichess_2000_pilot_lora` | 97% | 37% | 8% |

The adapter learned the JSON response shape, but chess move quality is weak.
Common failures include generic opening moves in unrelated positions and
malformed UCI-like strings.

## GGUF and Ollama Export

Unsloth successfully merged the adapter, but its GGUF wrapper failed on the
local llama.cpp checkout with:

```text
TypeError: ModelBase.__init__() got an unexpected keyword argument 'target_model_dir'
```

Workaround used:

1. Convert merged HF manually with `/home/pawelo/.unsloth/llama.cpp/convert_hf_to_gguf.py`.
2. Quantize manually with `/home/pawelo/.unsloth/llama.cpp/llama-quantize`.
3. Re-run `finetune.export_ollama --skip-export --template-family gemma4 --create-ollama`.

Final GGUF artifacts:

- `outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-q4_k_m.gguf` (~5.0 GB)
- `outputs/finetune/gemma4_e4b_lichess_2000_pilot_gguf/chess-ft-gemma4-e4b-pilot-q8_0.gguf` (~7.5 GB)

Temporary merged HF and BF16 GGUF files were removed after quantization.

## Ollama Held-Out Sanity

100 examples through Ollama API with `temperature=0`, `num_ctx=4096`,
`num_predict=24`, and `think=off`:

| Ollama model | JSON | Legal | Top-1 | Service errors |
|---|---:|---:|---:|---:|
| `chess-ft-gemma4-e4b-pilot-q4_k_m` | 100% | 39% | 5% | 0 |
| `chess-ft-gemma4-e4b-pilot-q8_0` | 100% | 35% | 6% | 0 |

Conclusion: the Gemma4 E4B pipeline works end to end, but this pilot is not a
strong chess model. Prefer the Qwen3.5 9B pilot for arena play unless Gemma is
being tested as an ablation.

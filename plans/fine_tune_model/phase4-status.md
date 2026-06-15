# Phase 4 Status - Merge to GGUF to Ollama

Status: completed for the Qwen3.5 9B 20k pilot
Date: 2026-06-12

## Summary

Phase 4 is complete for the trained pilot adapter:

- LoRA adapter:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_lora`
- GGUF output directory:
  `outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf`
- Ollama models:
  - `chess-ft-qwen35-9b-pilot-q4_k_m:latest`
  - `chess-ft-qwen35-9b-pilot-q8_0:latest`

The models use an Ollama template that reproduces the Qwen3.5 training chat
format with `enable_thinking=False`: user prompt, assistant turn, empty
`<think></think>` block, then the JSON move answer.

## Implemented Repo Changes

Added:

- `finetune/export_ollama.py`
  - Loads the trained LoRA adapter with Unsloth.
  - Exports GGUF files using Unsloth's llama.cpp workflow.
  - Writes one Ollama `Modelfile` per GGUF.
  - Optionally runs `ollama create` for each generated Modelfile.
  - Supports `--skip-export` so already-created GGUF files can be imported
    without repeating the expensive merge/conversion step.

Updated:

- `finetune/README.md`
  - Adds the Phase 4 export/import command and the held-out sanity-check
    command.

Generated adapters, merged weights, GGUF files, Modelfiles, and evaluation
outputs are intentionally kept under ignored `outputs/finetune/`.

## Export Notes

The first direct Unsloth export reached the merge step, but local llama.cpp
setup was blocked by missing system packages (`cmake` and libcurl development
headers). Since sudo was not available non-interactively, the local workaround
was:

1. install `cmake` and `ninja` into `.venv-train`;
2. clone llama.cpp under `$HOME/.unsloth/llama.cpp`;
3. build `llama-quantize` with curl disabled.

Build command:

```bash
PATH=.venv-train/bin:$PATH \
cmake -S "$HOME/.unsloth/llama.cpp" \
  -B "$HOME/.unsloth/llama.cpp/build" \
  -DGGML_CUDA=OFF \
  -DGGML_CURL=OFF \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF

PATH=.venv-train/bin:$PATH \
cmake --build "$HOME/.unsloth/llama.cpp/build" \
  --target llama-quantize \
  -j"$(nproc)"

cp "$HOME/.unsloth/llama.cpp/build/bin/llama-quantize" \
  "$HOME/.unsloth/llama.cpp/llama-quantize"
```

Manual conversion/quantization was then used from the merged HF output:

```bash
PATH=.venv-train/bin:$PATH \
.venv-train/bin/python "$HOME/.unsloth/llama.cpp/convert_hf_to_gguf.py" \
  outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf \
  --outfile outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.BF16.gguf \
  --outtype bf16

"$HOME/.unsloth/llama.cpp/llama-quantize" \
  outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.BF16.gguf \
  outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.Q4_K_M.gguf \
  Q4_K_M

"$HOME/.unsloth/llama.cpp/llama-quantize" \
  outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.BF16.gguf \
  outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.Q8_0.gguf \
  Q8_0
```

The intermediate BF16 GGUF and merged HF shards were deleted after Q4/Q8
quantization to recover disk space.

## Final GGUF Artifacts

```text
outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.Q4_K_M.gguf
outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf/qwen35_9b_lichess_2000_pilot.Q8_0.gguf
```

Approximate local file sizes:

```text
Q4_K_M: 5.4 GB
Q8_0:   9.2 GB
```

## Ollama Import

Modelfiles and Ollama models were created with:

```bash
.venv/bin/python -m finetune.export_ollama \
  --skip-export \
  --output-dir outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf \
  --model-name chess-ft-qwen35-9b-pilot \
  --create-ollama
```

Installed models:

```text
chess-ft-qwen35-9b-pilot-q4_k_m:latest    5.8 GB
chess-ft-qwen35-9b-pilot-q8_0:latest      9.8 GB
```

Both imported models use:

```text
PARAMETER num_ctx 4096
PARAMETER num_predict 24
PARAMETER stop <|im_end|>
PARAMETER temperature 0
```

The Ollama template is:

```text
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

## Held-Out Ollama Sanity Check

Evaluated the imported models through the normal Ollama API on the first 100
examples from:

```text
data/finetune/lichess_2000_2013-12_pilot.val.jsonl
```

Command shape:

```bash
.venv/bin/python -m finetune.evaluate_baseline \
  --dataset data/finetune/lichess_2000_2013-12_pilot.val.jsonl \
  --model chess-ft-qwen35-9b-pilot-q8_0 \
  --limit 100 \
  --timeout-seconds 300 \
  --num-ctx 4096 \
  --num-predict 24 \
  --think off \
  --progress-every 20
```

Results:

| Model | Examples | JSON parse | Legal move | Top-1 match | Service errors |
|---|---:|---:|---:|---:|---:|
| `chess-ft-qwen35-9b-pilot-q4_k_m` | 100 | 1.000 | 0.900 | 0.190 | 0 |
| `chess-ft-qwen35-9b-pilot-q8_0` | 100 | 1.000 | 0.920 | 0.190 | 0 |

Conclusion:

- The Ollama/GGUF export preserved the important behavior from the trained
  adapter.
- Q8_0 is the better default for arena benchmarking if VRAM/RAM budget allows.
- Q4_K_M is good enough to benchmark as the smaller quantization ablation.

## Disk Cleanup

To finish the export on local disk, redownloadable caches and obsolete probe
outputs were removed:

- Hugging Face cache for `unsloth/Qwen3.5-9B`;
- `outputs/finetune/qwen35_9b_lichess_2000_pilot_probe_lora`;
- `outputs/finetune/qwen35_9b_lichess_2000_speed10_lora`;
- `outputs/finetune/qwen35_9b_lichess_2000_speed10_bs2_lora`;
- intermediate BF16 GGUF and merged HF shards after Q4/Q8 were created.

The final LoRA adapter, final Q4/Q8 GGUF files, and Ollama models were kept.

## Next Operational Step

Proceed to Phase 5: benchmark `chess-ft-qwen35-9b-pilot-q8_0` and
`chess-ft-qwen35-9b-pilot-q4_k_m` in the arena against the same Stockfish
beginner configuration used for the base model.

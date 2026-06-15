from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_ADAPTER_DIR = Path("outputs/finetune/qwen35_9b_lichess_2000_pilot_lora")
DEFAULT_OUTPUT_DIR = Path("outputs/finetune/qwen35_9b_lichess_2000_pilot_gguf")
DEFAULT_MODEL_NAME = "chess-ft-qwen35-9b-pilot"
DEFAULT_MAX_SEQ_LENGTH = 1536

QWEN35_TEMPLATE = """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

GEMMA4_TEMPLATE = """<bos><|turn>user
{{ .Prompt }}<turn|>
<|turn>model
"""

TEMPLATE_FAMILIES = {
    "qwen35": {
        "template": QWEN35_TEMPLATE,
        "stop": ["<|im_end|>"],
    },
    "gemma4": {
        "template": GEMMA4_TEMPLATE,
        "stop": ["<turn|>", "<eos>"],
    },
}


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_export:
        _export_gguf(
            adapter_dir=args.adapter_dir,
            output_dir=output_dir,
            quantization_methods=args.quantization_method,
            max_seq_length=args.max_seq_length,
        )

    modelfiles = _write_modelfiles(
        output_dir=output_dir,
        model_name=args.model_name,
        template_family=args.template_family,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    for modelfile in modelfiles:
        print(f"Wrote {modelfile}")

    if args.create_ollama:
        for modelfile in modelfiles:
            model = modelfile.name.removeprefix("Modelfile.")
            _run(["ollama", "create", model, "-f", str(modelfile)])


def _export_gguf(
    *,
    adapter_dir: Path,
    output_dir: Path,
    quantization_methods: list[str],
    max_seq_length: int,
) -> None:
    try:
        from unsloth import FastModel
    except ImportError as exc:
        raise SystemExit(
            "Export dependencies are missing. Activate .venv-train and run "
            '`pip install -e ".[train]"`.'
        ) from exc

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization_methods,
    )


def _write_modelfiles(
    *,
    output_dir: Path,
    model_name: str,
    template_family: str,
    num_ctx: int,
    num_predict: int,
) -> list[Path]:
    ggufs = sorted(output_dir.glob("*.gguf"))
    if not ggufs:
        raise SystemExit(f"No GGUF files found in {output_dir}")

    template_config = TEMPLATE_FAMILIES[template_family]
    modelfiles: list[Path] = []
    for gguf in ggufs:
        quantization = _quantization_suffix(gguf)
        ollama_name = f"{model_name}-{quantization}"
        modelfile = output_dir / f"Modelfile.{ollama_name}"
        modelfile.write_text(
            "\n".join(
                [
                    f"FROM {gguf.resolve()}",
                    'TEMPLATE """' + str(template_config["template"]) + '"""',
                    *[f'PARAMETER stop "{stop}"' for stop in template_config["stop"]],
                    "PARAMETER temperature 0",
                    f"PARAMETER num_ctx {num_ctx}",
                    f"PARAMETER num_predict {num_predict}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        modelfiles.append(modelfile)
    return modelfiles


def _quantization_suffix(path: Path) -> str:
    stem = path.stem.lower()
    for quantization in ("q8_0", "q4_k_m", "f16", "bf16", "f32"):
        if quantization in stem:
            return quantization
    return stem.replace(".", "-").replace("_", "-")


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained LoRA adapter to GGUF and create Ollama Modelfiles."
    )
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--template-family",
        choices=sorted(TEMPLATE_FAMILIES),
        default="qwen35",
        help="Ollama prompt wrapper matching the chat template used during training.",
    )
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--quantization-method",
        action="append",
        default=[],
        help="Unsloth GGUF quantization, e.g. q4_k_m or q8_0. Repeat for multiple outputs.",
    )
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=24)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--create-ollama", action="store_true")
    args = parser.parse_args()
    if not args.quantization_method:
        args.quantization_method = ["q4_k_m", "q8_0"]
    return args


if __name__ == "__main__":
    main()

"""Export merged LoRA MOSS model to ONNX (+ ORT transformers optimize + Extensions tokenizer)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ru_bench.onnx_export import export_onnx_step


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/lora_ru_e2"),
        help="PEFT adapter dir (merged into base before export). Use '' for base-only.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("exports/moss_ru_e2_step.onnx"),
    )
    p.add_argument(
        "--sample-audio",
        type=Path,
        default=None,
        help="WAV for tracing shapes. Default: first clip from asr_sample.json",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run onnxruntime.transformers.optimizer (default: on)",
    )
    p.add_argument(
        "--float16",
        action="store_true",
        help="Convert optimized graph to FP16 (GPU Tensor Cores)",
    )
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="Pass use_gpu=True to ORT transformers optimizer",
    )
    p.add_argument(
        "--tokenizer-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export HfJsonTokenizer ONNX via onnxruntime-extensions (default: on)",
    )
    args = p.parse_args()

    sample = args.sample_audio
    if sample is None:
        manifest = json.loads(Path("data/manifests/asr_sample.json").read_text(encoding="utf-8"))
        sample = Path(manifest[0]["audio_path"])

    ckpt = args.checkpoint_dir
    if str(ckpt) in {"", "none", "None"}:
        ckpt = None

    arts = export_onnx_step(
        checkpoint_dir=ckpt,
        output_path=args.output,
        sample_audio=sample,
        opset=args.opset,
        optimize=args.optimize,
        float16=args.float16,
        use_gpu=args.use_gpu,
        export_tokenizer=args.tokenizer_onnx,
    )
    for key, path in arts.items():
        size = path.stat().st_size if path.is_file() else 0
        data = path.with_name(path.name + ".data")
        extra = f" + {data.name} ({data.stat().st_size / 1e9:.2f} GB)" if data.is_file() else ""
        print(f"{key}: {path} ({size} bytes){extra}")


if __name__ == "__main__":
    main()

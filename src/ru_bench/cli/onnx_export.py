"""Export split MOSS ONNX with KV-cache (audio + embed + lm)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ru_bench.onnx_kv import export_onnx_kv


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/lora_ru_fmt3"),
        help="PEFT adapter dir, or '' for base-only",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/moss_ru_fmt3_kv"),
    )
    p.add_argument("--sample-audio", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--optimize-lm",
        action="store_true",
        help="ORT transformers optimize on lm.onnx (opt_level=0)",
    )
    p.add_argument("--float16", action="store_true")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument(
        "--tokenizer-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--gqa",
        action="store_true",
        help="Fuse decode LM attention to GroupQueryAttention (qwen3 optimizer)",
    )
    p.add_argument(
        "--attention-op",
        default=None,
        choices=("Attention", "MultiHeadAttention", "GroupQueryAttention"),
    )
    p.add_argument(
        "--model-type",
        default="gpt2",
        help="ORT transformers optimizer model_type (use qwen3 with --gqa)",
    )
    args = p.parse_args()

    attention_op = args.attention_op
    model_type = args.model_type
    if args.gqa:
        attention_op = attention_op or "GroupQueryAttention"
        model_type = "qwen3" if model_type == "gpt2" else model_type

    sample = args.sample_audio
    if sample is None:
        z = Path("Звонок.wav")
        if z.is_file():
            sample = z
        else:
            manifest = json.loads(
                Path("data/manifests/asr_sample.json").read_text(encoding="utf-8")
            )
            sample = Path(manifest[0]["audio_path"])

    ckpt = args.checkpoint_dir
    if str(ckpt) in {"", "none", "None"}:
        ckpt = None

    arts = export_onnx_kv(
        checkpoint_dir=ckpt,
        output_dir=args.output_dir,
        sample_audio=sample,
        opset=args.opset,
        optimize_lm=args.optimize_lm,
        float16=args.float16,
        use_gpu=args.use_gpu,
        export_tokenizer=args.tokenizer_onnx,
        attention_op=attention_op,
        model_type=model_type,
    )
    for key, path in arts.items():
        if path.is_dir():
            print(f"{key}: {path}/")
            continue
        size = path.stat().st_size if path.is_file() else 0
        data = path.with_name(path.name + ".data")
        extra = (
            f" + {data.name} ({data.stat().st_size / 1e9:.2f} GB)"
            if data.is_file()
            else ""
        )
        print(f"{key}: {path} ({size} bytes){extra}")


if __name__ == "__main__":
    main()

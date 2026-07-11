"""Patch audio.onnx Whisper pos-embed for variable mel_len (10s chunks)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ru_bench.onnx_audio_dynpos import patch_audio_onnx_dynamic_pos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        default=Path("exports/moss_ru_fmt3_kv/audio.onnx"),
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Default: <src_dir>/audio.dynpos.onnx",
    )
    args = p.parse_args()
    dst = args.dst or (args.src.parent / "audio.dynpos.onnx")
    out = patch_audio_onnx_dynamic_pos(args.src, dst)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

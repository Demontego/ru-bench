"""Quantize MOSS one-step ONNX for deploy.

Methods:
  dynamic — CPU INT8 (quantize_dynamic, MatMul/Gemm). Best for CPUExecutionProvider.
  nbits   — weight-only MatMulNBits Q4/Q8 (GPU or CPU; size-first).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ru_bench.onnx_export import quantize_dynamic_int8, quantize_matmul_nbits


def _print_done(path: Path) -> None:
    data = path.with_name(path.name + ".data")
    size = path.stat().st_size
    extra = f" + {data.name} ({data.stat().st_size / 1e9:.2f} GB)" if data.is_file() else ""
    print(f"done: {path} ({size} bytes){extra}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("exports/moss_ru_fmt3_step.opt.onnx"),
        help="FP32 ONNX (prefer *.opt.onnx)",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--method",
        choices=("dynamic", "nbits"),
        default="dynamic",
        help="dynamic=CPU INT8 (default); nbits=weight-only Q4/Q8",
    )
    # nbits
    p.add_argument("--bits", type=int, choices=(4, 8), default=4)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--accuracy-level", type=int, default=4)
    p.add_argument("--symmetric", action="store_true")
    # dynamic
    p.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--reduce-range",
        action="store_true",
        help="7-bit weights (older CPUs without VNNI)",
    )
    p.add_argument(
        "--exclude-lm-head",
        action="store_true",
        help="Skip quantizing lm_head MatMul (often helps ASR token quality)",
    )
    args = p.parse_args()

    inp = args.input
    if not inp.is_file():
        raise SystemExit(f"missing input: {inp}")

    exclude: list[str] | None = None
    if args.exclude_lm_head:
        exclude = ["/model/lm_head/MatMul"]

    if args.method == "dynamic":
        out = args.output or inp.with_name(f"{inp.stem}.dynint8.onnx")
        print(
            f"dynamic INT8 {inp} -> {out} per_channel={args.per_channel} "
            f"reduce_range={args.reduce_range} exclude={exclude}",
            flush=True,
        )
        path = quantize_dynamic_int8(
            inp,
            out,
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            nodes_to_exclude=exclude,
        )
    else:
        out = args.output or inp.with_name(f"{inp.stem}.q{args.bits}.onnx")
        acc = None if args.accuracy_level < 0 else args.accuracy_level
        print(
            f"nbits {inp} -> {out} bits={args.bits} block={args.block_size} "
            f"accuracy_level={acc}",
            flush=True,
        )
        path = quantize_matmul_nbits(
            inp,
            out,
            bits=args.bits,
            block_size=args.block_size,
            is_symmetric=args.symmetric,
            accuracy_level=acc,
        )
    _print_done(path)


if __name__ == "__main__":
    main()

"""One-off: run LoRA checkpoint on Звонок.wav."""

from __future__ import annotations

import argparse
from pathlib import Path

from ru_bench.model_runner import load_model, run_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints/lora_ru_fmt")
    p.add_argument("--audio", default="Звонок.wav")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.is_file():
        raise SystemExit(f"audio not found: {audio}")

    print(f"Loading {args.checkpoint_dir} ...", flush=True)
    model, processor, device, dtype = load_model(adapter_path=args.checkpoint_dir)
    print(f"device={device} dtype={dtype}", flush=True)

    result = run_one(
        model, processor, device, dtype, str(audio), max_new_tokens=args.max_new_tokens
    )
    print("--- raw ---", flush=True)
    print(result["raw_text"], flush=True)
    print("--- segments ---", flush=True)
    for s in result["segments"]:
        print(
            f"[{s['start']:.2f}-{s['end']:.2f}][{s['speaker']}] {s['text']}",
            flush=True,
        )
    print("--- merged ---", flush=True)
    print(result["merged_text"], flush=True)


if __name__ == "__main__":
    main()

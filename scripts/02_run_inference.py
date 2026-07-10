import argparse
import json

from ru_bench import config
from ru_bench.data import load_manifest
from ru_bench.model_runner import load_model, run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    clips = load_manifest()
    if args.limit is not None:
        clips = clips[: args.limit]

    config.ASR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pending = [c for c in clips if not (config.ASR_RESULTS_DIR / f"{c.clip_id}.json").exists()]
    print(f"{len(clips) - len(pending)} already done, {len(pending)} to run")
    if not pending:
        return

    print("Loading model on CUDA...")
    model, processor, device, dtype = load_model()
    if device.type != "cuda":
        raise SystemExit(f"refusing non-CUDA device={device}")
    print(f"Model loaded on device={device}, dtype={dtype}")

    for i, clip in enumerate(pending):
        out_path = config.ASR_RESULTS_DIR / f"{clip.clip_id}.json"
        print(f"[{i + 1}/{len(pending)}] {clip.clip_id} ({clip.duration:.1f}s)")
        result = run_one(model, processor, device, dtype, clip.audio_path, config.MAX_NEW_TOKENS_ASR)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

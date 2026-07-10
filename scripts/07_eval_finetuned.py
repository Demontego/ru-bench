import argparse
import json
from pathlib import Path

from ru_bench import config
from ru_bench.data import load_manifest
from ru_bench.metrics_asr import compute_asr_metrics, save_metrics
from ru_bench.model_runner import load_model, run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--results-dir",
        default=str(config.FINETUNED_ASR_RESULTS_DIR),
        help="Per-clip JSON output dir (use separate dirs for different checkpoints)",
    )
    parser.add_argument(
        "--metrics-path",
        default=str(config.FINETUNED_ASR_METRICS_PATH),
        help="Where to write aggregate CER/WER JSON",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_path = Path(args.metrics_path)

    clips = load_manifest()
    if args.limit is not None:
        clips = clips[: args.limit]

    results_dir.mkdir(parents=True, exist_ok=True)
    pending = [c for c in clips if not (results_dir / f"{c.clip_id}.json").exists()]
    print(f"{len(clips) - len(pending)} already done, {len(pending)} to run")
    print(f"checkpoint={args.checkpoint_dir}")
    print(f"results_dir={results_dir}")

    if pending:
        print(f"Loading base model + LoRA adapter from {args.checkpoint_dir} on CUDA...")
        model, processor, device, dtype = load_model(adapter_path=args.checkpoint_dir)
        if device.type != "cuda":
            raise SystemExit(f"refusing non-CUDA device={device}")
        print(f"Model loaded on device={device}, dtype={dtype}")

        for i, clip in enumerate(pending):
            out_path = results_dir / f"{clip.clip_id}.json"
            print(f"[{i + 1}/{len(pending)}] {clip.clip_id} ({clip.duration:.1f}s)")
            result = run_one(model, processor, device, dtype, clip.audio_path, config.MAX_NEW_TOKENS_ASR)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    metrics = compute_asr_metrics(clips, results_dir=results_dir)
    save_metrics(metrics, path=metrics_path)
    print(f"Fine-tuned: n={metrics['n']} cer={metrics['cer']:.3f} wer={metrics['wer']:.3f}")
    print(f"Wrote {metrics_path}")

    if config.ASR_METRICS_PATH.exists():
        with open(config.ASR_METRICS_PATH, encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"Baseline:   n={baseline['n']} cer={baseline['cer']:.3f} wer={baseline['wer']:.3f}")


if __name__ == "__main__":
    main()

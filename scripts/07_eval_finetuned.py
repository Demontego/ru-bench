import argparse
import json

from ru_bench import config
from ru_bench.data import load_manifest
from ru_bench.metrics_asr import compute_asr_metrics, save_metrics
from ru_bench.model_runner import load_model, run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    clips = load_manifest()
    if args.limit is not None:
        clips = clips[: args.limit]

    config.FINETUNED_ASR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pending = [c for c in clips if not (config.FINETUNED_ASR_RESULTS_DIR / f"{c.clip_id}.json").exists()]
    print(f"{len(clips) - len(pending)} already done, {len(pending)} to run")

    if pending:
        print(f"Loading base model + LoRA adapter from {args.checkpoint_dir}...")
        model, processor, device, dtype = load_model(adapter_path=args.checkpoint_dir)
        print(f"Model loaded on device={device}, dtype={dtype}")

        for i, clip in enumerate(pending):
            out_path = config.FINETUNED_ASR_RESULTS_DIR / f"{clip.clip_id}.json"
            print(f"[{i + 1}/{len(pending)}] {clip.clip_id} ({clip.duration:.1f}s)")
            result = run_one(model, processor, device, dtype, clip.audio_path, config.MAX_NEW_TOKENS_ASR)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    metrics = compute_asr_metrics(clips, results_dir=config.FINETUNED_ASR_RESULTS_DIR)
    save_metrics(metrics, path=config.FINETUNED_ASR_METRICS_PATH)
    print(f"Fine-tuned: n={metrics['n']} cer={metrics['cer']:.3f} wer={metrics['wer']:.3f}")

    if config.ASR_METRICS_PATH.exists():
        with open(config.ASR_METRICS_PATH, encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"Baseline:   n={baseline['n']} cer={baseline['cer']:.3f} wer={baseline['wer']:.3f}")


if __name__ == "__main__":
    main()

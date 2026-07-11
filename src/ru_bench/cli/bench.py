"""Golos benchmark commands."""

from __future__ import annotations

import argparse
import json

from ru_bench import config
from ru_bench.data import (
    download_golos_test_tar,
    extract_golos_test_tar,
    load_all_manifests,
    load_manifest,
    save_manifest,
    select_asr_sample,
)
from ru_bench.metrics_asr import compute_asr_metrics, save_metrics
from ru_bench.model_runner import load_model, run_one
from ru_bench.report import main as report_main


def cmd_download(args: argparse.Namespace) -> None:
    print("Downloading Golos test.tar (cached if already present)...")
    tar_path = download_golos_test_tar()
    print(f"Extracting {tar_path}...")
    extract_dir = extract_golos_test_tar(tar_path)
    clips = load_all_manifests(extract_dir)
    print(f"Loaded {len(clips)} clips from manifests ({config.GOLOS_DOMAINS})")
    sample = select_asr_sample(clips, args.asr_n)
    save_manifest(sample)
    print(f"Sampled {len(sample)} clips -> {config.ASR_MANIFEST_PATH}")


def cmd_infer(args: argparse.Namespace) -> None:
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


def cmd_metrics(_args: argparse.Namespace) -> None:
    metrics = compute_asr_metrics()
    save_metrics(metrics)
    print(f"n={metrics['n']} cer={metrics['cer']:.3f} wer={metrics['wer']:.3f}")


def cmd_report(_args: argparse.Namespace) -> None:
    report_main()


def dispatch(argv: list[str]) -> None:
    """Handle ``bench <action> [options]``."""
    import argparse

    if not argv or argv[0] in {"-h", "--help"}:
        print("ru-bench bench <download|infer|metrics|report> [options]")
        return

    parser = argparse.ArgumentParser(prog="ru-bench bench")
    actions = parser.add_subparsers(dest="action", required=True)

    p = actions.add_parser("download")
    p.add_argument("--asr-n", type=int, default=config.ASR_SAMPLE_SIZE)
    p.set_defaults(func=cmd_download)

    p = actions.add_parser("infer")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_infer)

    p = actions.add_parser("metrics")
    p.set_defaults(func=cmd_metrics)

    p = actions.add_parser("report")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)

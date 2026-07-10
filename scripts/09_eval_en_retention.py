"""EN retention eval: base MOSS vs LoRA FT on EN subset of a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ru_bench import config
from ru_bench.data import ClipRef, load_manifest
from ru_bench.metrics_asr import compute_asr_metrics, save_metrics
from ru_bench.model_runner import load_model, run_one

EN_DOMAINS = config.EN_DOMAINS


def filter_en(clips: list[ClipRef]) -> list[ClipRef]:
    return [c for c in clips if c.domain in EN_DOMAINS]


def run_inference(
    clips: list[ClipRef],
    results_dir: Path,
    adapter_path: str | None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pending = [c for c in clips if not (results_dir / f"{c.clip_id}.json").exists()]
    print(f"{len(clips) - len(pending)} already done, {len(pending)} to run")
    print(f"adapter={adapter_path!r} results_dir={results_dir}")
    if not pending:
        return

    label = "base" if adapter_path is None else f"adapter={adapter_path}"
    print(f"Loading model ({label}) on CUDA...")
    model, processor, device, dtype = load_model(adapter_path=adapter_path)
    if device.type != "cuda":
        raise SystemExit(f"refusing non-CUDA device={device}")
    print(f"Model loaded on device={device}, dtype={dtype}")

    for i, clip in enumerate(pending):
        out_path = results_dir / f"{clip.clip_id}.json"
        print(f"[{i + 1}/{len(pending)}] {clip.clip_id} ({clip.domain} {clip.duration:.1f}s)")
        result = run_one(model, processor, device, dtype, clip.audio_path, config.MAX_NEW_TOKENS_ASR)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def print_examples(base_metrics: dict, ft_metrics: dict, n: int = 3) -> None:
    base_by_id = {p["clip_id"]: p for p in base_metrics["per_clip"]}
    ft_by_id = {p["clip_id"]: p for p in ft_metrics["per_clip"]}
    common = [cid for cid in base_by_id if cid in ft_by_id]
    # Prefer clips where both have non-empty hyp, pick mid/worst/best-ish spread
    scored = []
    for cid in common:
        b, f = base_by_id[cid], ft_by_id[cid]
        if not b["hypothesis"].strip() and not f["hypothesis"].strip():
            continue
        scored.append((abs(f["cer"] - b["cer"]), cid))
    scored.sort(reverse=True)
    pick = [cid for _, cid in scored[:n]]
    if len(pick) < n:
        pick = common[:n]
    print("\n=== PRED/REF examples (base vs FT) ===")
    for cid in pick:
        b, f = base_by_id[cid], ft_by_id[cid]
        print(f"\n--- {cid} ---")
        print(f"REF:  {b['reference']}")
        print(f"BASE: {b['hypothesis']}  (cer={b['cer']:.3f})")
        print(f"FT:   {f['hypothesis']}  (cer={f['cer']:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="EN retention: base vs LoRA FT")
    parser.add_argument("--manifest", default=str(config.DEV_MANIFEST_PATH))
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Cap EN clips (after filter)")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--domains",
        default="fleurs_en,librispeech_clean",
        help="Comma-separated EN domains to include",
    )
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-ft", action="store_true")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    domains = {d.strip() for d in args.domains.split(",") if d.strip()}
    clips = [c for c in load_manifest(Path(args.manifest)) if c.domain in domains]
    clips = sorted(clips, key=lambda c: c.clip_id)
    if args.limit is not None:
        import random

        rng = random.Random(args.seed)
        clips = rng.sample(clips, min(args.limit, len(clips)))
        clips = sorted(clips, key=lambda c: c.clip_id)

    print(f"EN clips: n={len(clips)} domains={sorted(domains)} manifest={args.manifest}")
    print(f"total_audio_s={sum(c.duration for c in clips):.1f}")

    base_dir = config.RESULTS_DIR / "asr_en_base"
    ft_dir = config.RESULTS_DIR / "asr_en_ft"
    base_metrics_path = config.RESULTS_DIR / "metrics_asr_en_base.json"
    ft_metrics_path = config.RESULTS_DIR / "metrics_asr_en_ft.json"
    compare_path = config.RESULTS_DIR / "metrics_asr_en_compare.json"

    if not args.skip_base:
        run_inference(clips, base_dir, adapter_path=None)
    if not args.skip_ft:
        run_inference(clips, ft_dir, adapter_path=args.checkpoint_dir)

    base_m = compute_asr_metrics(clips, results_dir=base_dir)
    ft_m = compute_asr_metrics(clips, results_dir=ft_dir)
    save_metrics(base_m, path=base_metrics_path)
    save_metrics(ft_m, path=ft_metrics_path)

    delta_cer = ft_m["cer"] - base_m["cer"]
    delta_wer = ft_m["wer"] - base_m["wer"]
    if abs(delta_cer) < 0.01 and abs(delta_wer) < 0.02:
        verdict = "same"
    elif delta_cer > 0 or delta_wer > 0:
        verdict = "worse"
    else:
        verdict = "better"

    compare = {
        "n": min(base_m["n"], ft_m["n"]),
        "domains": sorted(domains),
        "manifest": args.manifest,
        "checkpoint": args.checkpoint_dir,
        "base": {"n": base_m["n"], "cer": base_m["cer"], "wer": base_m["wer"]},
        "finetuned": {"n": ft_m["n"], "cer": ft_m["cer"], "wer": ft_m["wer"]},
        "delta": {"cer": delta_cer, "wer": delta_wer},
        "verdict": verdict,
        "paths": {
            "base_results": str(base_dir),
            "ft_results": str(ft_dir),
            "base_metrics": str(base_metrics_path),
            "ft_metrics": str(ft_metrics_path),
        },
    }
    save_metrics(compare, path=compare_path)

    print("\n=== EN retention summary ===")
    print(f"BASE: n={base_m['n']} cer={base_m['cer']:.3f} wer={base_m['wer']:.3f}")
    print(f"FT:   n={ft_m['n']} cer={ft_m['cer']:.3f} wer={ft_m['wer']:.3f}")
    print(f"delta cer={delta_cer:+.3f} wer={delta_wer:+.3f} -> {verdict}")
    print(f"Wrote {compare_path}")

    if base_m["n"] and ft_m["n"]:
        print_examples(base_m, ft_m, n=args.examples)


if __name__ == "__main__":
    main()

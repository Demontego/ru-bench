"""Mixed RU+EN eval: stratified sample from dev_sample, base vs LoRA FT."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import jiwer

from ru_bench import config
from ru_bench.data import ClipRef, load_manifest
from ru_bench.metrics_asr import compute_asr_metrics, load_hypothesis, save_metrics
from ru_bench.model_runner import load_model, run_one


def stratified_mix(
    clips: list[ClipRef],
    n_ru: int,
    n_en: int,
    seed: int,
) -> list[ClipRef]:
    ru = [c for c in clips if c.domain in config.RU_DOMAINS]
    en = [c for c in clips if c.domain in config.EN_DOMAINS]
    rng = random.Random(seed)
    pick_ru = rng.sample(ru, min(n_ru, len(ru)))
    pick_en = rng.sample(en, min(n_en, len(en)))
    return sorted(pick_ru + pick_en, key=lambda c: c.clip_id)


def lang_of(clip: ClipRef) -> str:
    return "en" if clip.domain in config.EN_DOMAINS else "ru"


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
        result = run_one(
            model, processor, device, dtype, clip.audio_path, config.MAX_NEW_TOKENS_ASR
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def reuse_en_preds(clips: list[ClipRef], src_dir: Path, dst_dir: Path) -> int:
    """Copy existing EN preds into mix results dir. Returns count copied/linked."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for clip in clips:
        if lang_of(clip) != "en":
            continue
        src = src_dir / f"{clip.clip_id}.json"
        dst = dst_dir / f"{clip.clip_id}.json"
        if dst.exists():
            n += 1
            continue
        if not src.exists():
            continue
        shutil.copy2(src, dst)
        n += 1
    return n


def metrics_by_lang(clips: list[ClipRef], results_dir: Path) -> dict:
    overall = compute_asr_metrics(clips, results_dir=results_dir)
    by_lang: dict[str, list[ClipRef]] = defaultdict(list)
    for c in clips:
        if (results_dir / f"{c.clip_id}.json").exists():
            by_lang[lang_of(c)].append(c)
    per_lang = {}
    for lang, subset in sorted(by_lang.items()):
        m = compute_asr_metrics(subset, results_dir=results_dir)
        per_lang[lang] = {"n": m["n"], "cer": m["cer"], "wer": m["wer"]}
    return {
        "n": overall["n"],
        "cer": overall["cer"],
        "wer": overall["wer"],
        "per_lang": per_lang,
        "per_clip": overall["per_clip"],
    }


def casefold_metrics(clips: list[ClipRef], results_dir: Path) -> dict:
    clips = [c for c in clips if (results_dir / f"{c.clip_id}.json").exists()]
    refs = [c.text.lower() for c in clips]
    hyps = [load_hypothesis(c, results_dir).lower() for c in clips]
    if not refs:
        return {"n": 0, "cer": float("nan"), "wer": float("nan")}
    return {
        "n": len(refs),
        "cer": jiwer.process_characters(refs, hyps).cer,
        "wer": jiwer.process_words(refs, hyps).wer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed RU+EN ASR: base vs LoRA FT")
    parser.add_argument("--manifest", default=str(config.DEV_MANIFEST_PATH))
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--n-ru", type=int, default=100)
    parser.add_argument("--n-en", type=int, default=100)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-ft", action="store_true")
    args = parser.parse_args()

    all_clips = load_manifest(Path(args.manifest))
    clips = stratified_mix(all_clips, args.n_ru, args.n_en, args.seed)
    ru_clips = [c for c in clips if lang_of(c) == "ru"]
    en_clips = [c for c in clips if lang_of(c) == "en"]
    print(
        f"mix n={len(clips)} ru={len(ru_clips)} en={len(en_clips)} "
        f"manifest={args.manifest} seed={args.seed}"
    )
    print(f"total_audio_s={sum(c.duration for c in clips):.1f}")

    base_dir = config.RESULTS_DIR / "asr_mix_base"
    ft_dir = config.RESULTS_DIR / "asr_mix_ft"
    base_metrics_path = config.RESULTS_DIR / "metrics_asr_mix_base.json"
    ft_metrics_path = config.RESULTS_DIR / "metrics_asr_mix_ft.json"
    compare_path = config.RESULTS_DIR / "metrics_asr_mix_compare.json"

    # Reuse EN retention preds; only RU needs new inference.
    n_en_base = reuse_en_preds(clips, config.RESULTS_DIR / "asr_en_base", base_dir)
    n_en_ft = reuse_en_preds(clips, config.RESULTS_DIR / "asr_en_ft", ft_dir)
    print(f"reused EN preds: base={n_en_base} ft={n_en_ft}")

    if not args.skip_base:
        run_inference(ru_clips, base_dir, adapter_path=None)
    if not args.skip_ft:
        run_inference(ru_clips, ft_dir, adapter_path=args.checkpoint_dir)

    base_m = metrics_by_lang(clips, base_dir)
    ft_m = metrics_by_lang(clips, ft_dir)
    save_metrics(base_m, path=base_metrics_path)
    save_metrics(ft_m, path=ft_metrics_path)

    base_ci = casefold_metrics(clips, base_dir)
    ft_ci = casefold_metrics(clips, ft_dir)
    en_only = [c for c in clips if lang_of(c) == "en"]
    base_en_ci = casefold_metrics(en_only, base_dir)
    ft_en_ci = casefold_metrics(en_only, ft_dir)

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
        "n_ru": args.n_ru,
        "n_en": args.n_en,
        "seed": args.seed,
        "manifest": args.manifest,
        "checkpoint": args.checkpoint_dir,
        "base": {
            "n": base_m["n"],
            "cer": base_m["cer"],
            "wer": base_m["wer"],
            "per_lang": base_m["per_lang"],
            "casefold": base_ci,
            "en_casefold": base_en_ci,
        },
        "finetuned": {
            "n": ft_m["n"],
            "cer": ft_m["cer"],
            "wer": ft_m["wer"],
            "per_lang": ft_m["per_lang"],
            "casefold": ft_ci,
            "en_casefold": ft_en_ci,
        },
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

    print("\n=== MIX summary ===")
    print(f"BASE: n={base_m['n']} cer={base_m['cer']:.3f} wer={base_m['wer']:.3f} per_lang={base_m['per_lang']}")
    print(f"FT:   n={ft_m['n']} cer={ft_m['cer']:.3f} wer={ft_m['wer']:.3f} per_lang={ft_m['per_lang']}")
    print(f"delta cer={delta_cer:+.3f} wer={delta_wer:+.3f} -> {verdict}")
    print(f"casefold overall base={base_ci} ft={ft_ci}")
    print(f"Wrote {compare_path}")


if __name__ == "__main__":
    main()

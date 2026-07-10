"""Build train/dev manifests from Golos CDN + open-source HF corpora.

Policy (default):
  - Eval = 10% of pool, prefer ~50/50 RU/EN (fair bilingual metrics)
  - Train = all remaining (RU-heavy OK; modest EN for retention)
  - No huge EN download just to balance train

Examples:
  # Mixed default: farfield + golos10h + FLEURS RU/EN
  uv run python scripts/05_download_train_data.py --sources mixed

  # Golos farfield only (~15.4GB tar)
  uv run python scripts/05_download_train_data.py --sources golos_farfield

  # HF-only smoke (no 15GB farfield)
  uv run python scripts/05_download_train_data.py --sources golos10h,fleurs_ru,fleurs_en \\
      --n-train 500 --n-dev 50
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from ru_bench import config
from ru_bench.hf_sources import load_opensource_sources
from ru_bench.train_data import (
    clip_language,
    download_train_crowd_shard,
    download_train_farfield,
    ensure_golos_train_manifests,
    filter_existing_audio,
    load_clips_from_manifests,
    save_clip_manifest,
    select_train_dev_split,
    select_train_dev_split_ru_primary,
)


def _parse_sources(raw: str) -> list[str]:
    raw = raw.strip().lower()
    if raw in ("mixed", "default", "all"):
        return list(config.DEFAULT_TRAIN_SOURCES)
    if raw in ("hf", "opensource"):
        return [s for s in config.DEFAULT_TRAIN_SOURCES if s != "golos_farfield"]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_golos_cdn(sources: list[str], crowd_shard: int) -> list:
    clips = []
    need_manifests = "golos_farfield" in sources or "golos_crowd" in sources
    if need_manifests:
        # Farfield/crowd audio tars omit transcripts; manifests are in crowd9.
        ensure_golos_train_manifests()

    if "golos_farfield" in sources:
        print("Downloading Golos train_farfield.tar (~15.4GB, cached if present)...")
        extract_dir = download_train_farfield()
        limit = config.SOURCE_LIMITS.get("golos_farfield")
        farfield = load_clips_from_manifests(
            extract_dir, domain_filter="farfield", limit=limit
        )
        print(f"  golos_farfield: {len(farfield)} clips")
        clips.extend(farfield)
    if "golos_crowd" in sources:
        print(f"Downloading Golos train_crowd{crowd_shard}.tar...")
        extract_dir = download_train_crowd_shard(crowd_shard)
        crowd = load_clips_from_manifests(extract_dir, domain_filter="crowd")
        print(f"  golos_crowd{crowd_shard}: {len(crowd)} clips")
        clips.extend(crowd)
    return clips


def _lang_counts(clips: list) -> dict[str, int]:
    return dict(Counter(clip_language(c) for c in clips))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--sources",
        default="mixed",
        help=(
            "Comma list or alias: mixed|hf|golos_farfield|golos_crowd|"
            "golos10h|fleurs_ru|cv_ru|fleurs_en|librispeech_clean|synth_diar_ru"
        ),
    )
    parser.add_argument(
        "--domain",
        choices=["farfield", "crowd", "mixed"],
        default=None,
        help="Legacy: farfield/crowd map to --sources; prefer --sources",
    )
    parser.add_argument("--crowd-shard", type=int, default=0, help="shard 0-9 for golos_crowd")
    parser.add_argument(
        "--n-train",
        type=int,
        default=config.TRAIN_SAMPLE_SIZE,
        help="0 = use full balanced pool (default); >0 = legacy absolute train cap",
    )
    parser.add_argument(
        "--n-dev",
        type=int,
        default=config.DEV_SAMPLE_SIZE,
        help="0 with --n-train 0 = EVAL_RATIO stratified split; >0 = legacy absolute dev size",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=config.TRAIN_MAX_AUDIO_SECONDS,
        help="drop clips longer than this (VRAM safety)",
    )
    args = parser.parse_args()

    if args.domain == "farfield":
        sources = ["golos_farfield"]
    elif args.domain == "crowd":
        sources = ["golos_crowd"]
    else:
        sources = _parse_sources(args.sources)

    print(f"Sources: {sources}")
    clips = []
    clips.extend(_load_golos_cdn(sources, args.crowd_shard))
    hf_sources = [s for s in sources if s in {
        "golos10h", "fleurs_ru", "cv_ru", "fleurs_en", "librispeech_clean", "synth_diar_ru"
    }]
    if hf_sources:
        print("Materializing HF opensource sources (streaming, cached wavs)...")
        clips.extend(load_opensource_sources(hf_sources))

    if not clips:
        raise SystemExit("No clips loaded from any source")

    before = len(clips)
    if args.max_audio_seconds and args.max_audio_seconds > 0:
        clips = [c for c in clips if c.duration <= args.max_audio_seconds]
    print(f"Pool: {before} -> {len(clips)} after max_audio_seconds={args.max_audio_seconds}")
    print("By domain:", dict(Counter(c.domain for c in clips)))
    print("By language:", _lang_counts(clips))

    clips, missing = filter_existing_audio(clips)
    print(f"Audio validation: kept={len(clips)} missing={missing}")
    if missing:
        print(f"WARNING: dropped {missing} clips with missing audio paths")
    if not clips:
        raise SystemExit("No clips left after audio validation")

    if args.n_train <= 0 and args.n_dev <= 0:
        train, dev = select_train_dev_split_ru_primary(clips)
    else:
        train, dev = select_train_dev_split(clips, args.n_train, args.n_dev)

    train, miss_train = filter_existing_audio(train)
    dev, miss_dev = filter_existing_audio(dev)
    if miss_train or miss_dev:
        raise SystemExit(
            f"Audio missing after split: train_missing={miss_train} dev_missing={miss_dev}"
        )

    save_clip_manifest(train, config.TRAIN_MANIFEST_PATH)
    save_clip_manifest(dev, config.DEV_MANIFEST_PATH)

    train_lang = _lang_counts(train)
    dev_lang = _lang_counts(dev)
    print(f"Train: {len(train)} clips -> {config.TRAIN_MANIFEST_PATH}")
    print(f"  train_ru={train_lang.get('ru', 0)} train_en={train_lang.get('en', 0)}")
    print(f"Dev:   {len(dev)} clips -> {config.DEV_MANIFEST_PATH}")
    print(f"  eval_ru={dev_lang.get('ru', 0)} eval_en={dev_lang.get('en', 0)}")
    print(f"Total: {len(train) + len(dev)} (train+eval), missing=0")
    print("Train domains:", dict(Counter(c.domain for c in train)))
    print("Dev domains:", dict(Counter(c.domain for c in dev)))


if __name__ == "__main__":
    main()

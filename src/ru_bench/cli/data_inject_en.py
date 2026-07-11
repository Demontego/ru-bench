"""Inject LibriConvo EN multi-spk diar into existing train/dev manifests.

Avoids full Golos/farfield rebuild. Materializes ``libri_convo_en``, splits
90/10, appends (dedupe by clip_id).

  uv run python scripts/25_inject_en_diar.py --limit 1500
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path

from ru_bench import config
from ru_bench.data import ClipRef, load_manifest
from ru_bench.hf_sources import load_libri_convo_en
from ru_bench.train_data import clip_language, filter_existing_audio, save_clip_manifest

logger = logging.getLogger(__name__)


def _merge(base: list[ClipRef], extra: list[ClipRef]) -> list[ClipRef]:
    seen = {c.clip_id for c in base}
    out = list(base)
    for c in extra:
        if c.clip_id in seen:
            continue
        seen.add(c.clip_id)
        out.append(c)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=config.SOURCE_LIMITS["libri_convo_en"])
    p.add_argument("--eval-ratio", type=float, default=config.EVAL_RATIO)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--train-manifest", type=Path, default=config.TRAIN_MANIFEST_PATH)
    p.add_argument("--dev-manifest", type=Path, default=config.DEV_MANIFEST_PATH)
    args = p.parse_args()

    print(f"Materializing libri_convo_en limit={args.limit}...", flush=True)
    en = load_libri_convo_en(args.limit)
    en, missing = filter_existing_audio(en)
    if missing:
        print(f"WARNING: dropped {missing} EN diar clips with missing audio", flush=True)
    if not en:
        raise SystemExit("no libri_convo_en clips materialized")

    rng = random.Random(args.seed)
    rng.shuffle(en)
    n_dev = max(1, int(round(len(en) * args.eval_ratio)))
    n_dev = min(n_dev, len(en) - 1) if len(en) > 1 else len(en)
    en_dev, en_train = en[:n_dev], en[n_dev:]

    train = load_manifest(args.train_manifest) if args.train_manifest.exists() else []
    dev = load_manifest(args.dev_manifest) if args.dev_manifest.exists() else []
    # Drop stale libri_convo rows so re-run is idempotent with fresh pool.
    train = [c for c in train if c.domain != "libri_convo_en"]
    dev = [c for c in dev if c.domain != "libri_convo_en"]

    train = _merge(train, en_train)
    dev = _merge(dev, en_dev)
    save_clip_manifest(train, args.train_manifest)
    save_clip_manifest(dev, args.dev_manifest)

    print(
        f"Injected EN diar: train+={len(en_train)} dev+={len(en_dev)} "
        f"(total train={len(train)} dev={len(dev)})",
        flush=True,
    )
    print("Train langs:", dict(Counter(clip_language(c) for c in train)), flush=True)
    print("Dev langs:", dict(Counter(clip_language(c) for c in dev)), flush=True)
    print(
        "Train domains:",
        dict(Counter(c.domain for c in train)),
        flush=True,
    )
    print(
        "Dev domains:",
        dict(Counter(c.domain for c in dev)),
        flush=True,
    )


if __name__ == "__main__":
    main()

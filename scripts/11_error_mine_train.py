"""Error-mine Golos FT preds -> supplemental / upsampled train manifest.

Reads results/metrics_asr_finetuned.json (and optional EN FT metrics),
classifies hard clips (empty hyp / high CER), finds similar *train* examples
(same style: short commands, names, code-switch, farfield), and writes an
extended train manifest with modest EN retention (~10-15%).

Does NOT touch dev_sample.json or asr_sample.json. Never puts eval/test
clip_ids into the new train set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ru_bench import config
from ru_bench.data import ClipRef, load_manifest
from ru_bench.train_data import clip_language, filter_existing_audio, save_clip_manifest

WAKE = ("салют", "джой", "афина", "сбер", "алиса", "ок гугл")
LATIN_RE = re.compile(r"[a-zA-Z]")
# rough RU name-ish: 1–3 tokens, no wake, no digits-as-words heavy cmds
DIGIT_WORDS = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "сто",
    "тысяч",
)


def _words(text: str) -> list[str]:
    return [w for w in text.lower().split() if w]


def classify_text(text: str) -> set[str]:
    tags: set[str] = set()
    ws = _words(text)
    low = text.lower()
    if any(low.startswith(w) or f" {w} " in f" {low} " for w in WAKE):
        tags.add("wake_cmd")
    if len(ws) <= 3:
        tags.add("short")
    if LATIN_RE.search(text):
        tags.add("code_switch")
    # translit / brand-ish latin tokens often appear as RU spelling of EN
    if any(t in low for t in ("эйч ди", "ютьюб", "ютуб", "бридж", "гуд лайф", "хд")):
        tags.add("code_switch")
    if any(d in low for d in DIGIT_WORDS) or any(ch.isdigit() for ch in text):
        tags.add("numbers")
    # name-like: short, no wake, looks like person/place
    if 1 <= len(ws) <= 3 and "wake_cmd" not in tags and "numbers" not in tags:
        tags.add("name_like")
    if not tags:
        tags.add("other")
    return tags


def load_hard_rows(
    metrics_path: Path,
    *,
    cer_threshold: float,
) -> list[dict]:
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    hard = []
    for row in metrics.get("per_clip", []):
        hyp = (row.get("hypothesis") or "").strip()
        cer = float(row.get("cer", 0.0))
        if not hyp or cer >= cer_threshold:
            hard.append(row)
    return hard


def mine_error_tags(hard_rows: list[dict]) -> Counter:
    c: Counter = Counter()
    for row in hard_rows:
        tags = classify_text(row.get("reference") or "")
        if not (row.get("hypothesis") or "").strip():
            tags = set(tags) | {"empty_hyp"}
        for t in tags:
            c[t] += 1
    return c


def select_similar(
    pool: list[ClipRef],
    target_tags: set[str],
    *,
    rng: random.Random,
    limit: int,
    exclude_ids: set[str],
) -> list[ClipRef]:
    scored: list[tuple[int, ClipRef]] = []
    for clip in pool:
        if clip.clip_id in exclude_ids:
            continue
        tags = classify_text(clip.text)
        overlap = len(tags & target_tags)
        if overlap == 0:
            continue
        # prefer farfield + more tag overlap
        bonus = 1 if clip.domain == "farfield" else 0
        scored.append((overlap + bonus, clip))
    scored.sort(key=lambda x: (-x[0], x[1].clip_id))
    # take top matches then shuffle within score bands for diversity
    top = [c for _, c in scored[: max(limit * 3, limit)]]
    rng.shuffle(top)
    return top[:limit]


def build_error_focus_train(
    base_train: list[ClipRef],
    hard_rows: list[dict],
    *,
    exclude_ids: set[str],
    en_fraction: float,
    hard_upsample: int,
    hard_pool_per_tag: int,
    base_ru_keep: int | None,
    seed: int,
) -> tuple[list[ClipRef], dict]:
    rng = random.Random(seed)
    # Aggregate target tags from hard refs (ignore empty_hyp for matching)
    # "other" is too broad for similarity mining — skip it
    target_tags: set[str] = set()
    for row in hard_rows:
        target_tags |= classify_text(row.get("reference") or "") - {"empty_hyp", "other"}
    if not target_tags:
        target_tags = {"short", "wake_cmd", "name_like", "code_switch"}

    ru_pool = [c for c in base_train if clip_language(c) == "ru" and c.clip_id not in exclude_ids]
    en_pool = [c for c in base_train if clip_language(c) == "en" and c.clip_id not in exclude_ids]

    # Prefer farfield for hard-style mining
    similar: list[ClipRef] = []
    seen: set[str] = set()
    for tag in sorted(target_tags):
        picked = select_similar(
            ru_pool,
            {tag},
            rng=rng,
            limit=hard_pool_per_tag,
            exclude_ids=exclude_ids | seen,
        )
        for c in picked:
            if c.clip_id not in seen:
                seen.add(c.clip_id)
                similar.append(c)

    # Also pull extra farfield short/wake if available
    farfield_extra = select_similar(
        [c for c in ru_pool if c.domain == "farfield"],
        {"short", "wake_cmd", "name_like", "code_switch"},
        rng=rng,
        limit=hard_pool_per_tag,
        exclude_ids=exclude_ids | seen,
    )
    for c in farfield_extra:
        if c.clip_id not in seen:
            seen.add(c.clip_id)
            similar.append(c)

    # Upsample hard-style clips
    hard_block: list[ClipRef] = []
    for _ in range(hard_upsample):
        hard_block.extend(similar)

    # Base RU: keep most of original RU (or capped), minus those already in hard block
    rng.shuffle(ru_pool)
    if base_ru_keep is None:
        base_ru = [c for c in ru_pool if c.clip_id not in seen]
    else:
        base_ru = [c for c in ru_pool if c.clip_id not in seen][:base_ru_keep]

    # EN retention target
    ru_part = base_ru + hard_block
    n_en = int(round(len(ru_part) * en_fraction / max(1e-9, 1.0 - en_fraction)))
    n_en = min(n_en, len(en_pool))
    rng.shuffle(en_pool)
    en_part = en_pool[:n_en]

    out = ru_part + en_part
    rng.shuffle(out)
    out, missing = filter_existing_audio(out)
    if missing:
        print(f"WARNING: dropped {missing} clips with missing audio")

    stats = {
        "hard_rows": len(hard_rows),
        "target_tags": sorted(target_tags),
        "similar_unique": len(similar),
        "hard_upsample": hard_upsample,
        "hard_block": len(hard_block),
        "base_ru": len(base_ru),
        "en": len(en_part),
        "total": len(out),
        "en_pct": round(100.0 * len(en_part) / max(len(out), 1), 1),
        "domains": dict(Counter(c.domain for c in out)),
        "langs": dict(Counter(clip_language(c) for c in out)),
        "missing_audio": missing,
    }
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default=str(config.FINETUNED_ASR_METRICS_PATH),
        help="Golos FT metrics JSON with per_clip",
    )
    parser.add_argument(
        "--en-metrics",
        default="",
        help="optional EN FT metrics (empty hyp / high CER tags only)",
    )
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    parser.add_argument("--train-manifest", default=str(config.TRAIN_MANIFEST_PATH))
    parser.add_argument(
        "--out",
        default=str(config.DATA_DIR / "manifests" / "train_error_focus.json"),
    )
    parser.add_argument("--en-fraction", type=float, default=0.12, help="target EN share")
    parser.add_argument("--hard-upsample", type=int, default=3, help="repeat factor for hard-style")
    parser.add_argument("--hard-pool-per-tag", type=int, default=400)
    parser.add_argument(
        "--base-ru-keep",
        type=int,
        default=0,
        help="0 = keep all non-hard RU from base train; >0 = cap",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument(
        "--report",
        default=str(config.RESULTS_DIR / "error_mine_report.json"),
    )
    args = parser.parse_args()

    hard = load_hard_rows(Path(args.metrics), cer_threshold=args.cer_threshold)
    if args.en_metrics:
        en_path = Path(args.en_metrics)
        if en_path.exists():
            hard.extend(load_hard_rows(en_path, cer_threshold=args.cer_threshold))

    tag_counts = mine_error_tags(hard)
    print(f"Hard clips: {len(hard)}")
    print("Error tags:", dict(tag_counts))
    for row in sorted(hard, key=lambda r: -float(r.get("cer", 0)))[:20]:
        empty = not (row.get("hypothesis") or "").strip()
        ref = (row.get("reference") or "")[:80]
        print(f"  cer={row.get('cer', 0):.2f} empty={empty} tags={sorted(classify_text(ref))} | {ref}")

    # Exclude eval/test ids
    exclude: set[str] = set()
    for path in (config.ASR_MANIFEST_PATH, config.DEV_MANIFEST_PATH):
        if path.exists():
            exclude |= {c.clip_id for c in load_manifest(path)}
    print(f"Exclude eval/test ids: {len(exclude)}")

    base_train = load_manifest(Path(args.train_manifest))
    base_ru_keep = None if args.base_ru_keep <= 0 else args.base_ru_keep
    train, stats = build_error_focus_train(
        base_train,
        hard,
        exclude_ids=exclude,
        en_fraction=args.en_fraction,
        hard_upsample=args.hard_upsample,
        hard_pool_per_tag=args.hard_pool_per_tag,
        base_ru_keep=base_ru_keep,
        seed=args.seed,
    )

    out_path = Path(args.out)
    save_clip_manifest(train, out_path)
    report = {
        "hard_tag_counts": dict(tag_counts),
        "hard_examples": [
            {
                "clip_id": r.get("clip_id"),
                "cer": r.get("cer"),
                "empty": not (r.get("hypothesis") or "").strip(),
                "reference": r.get("reference"),
                "hypothesis": r.get("hypothesis"),
                "tags": sorted(classify_text(r.get("reference") or "")),
            }
            for r in sorted(hard, key=lambda x: -float(x.get("cer", 0)))
        ],
        "stats": stats,
        "out": str(out_path),
        "dev_untouched": str(config.DEV_MANIFEST_PATH),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train)} clips -> {out_path}")
    print("Stats:", json.dumps(stats, ensure_ascii=False))
    print(f"Report -> {report_path}")
    # silence unused import warning for asdict in some linters
    _ = asdict


if __name__ == "__main__":
    main()

"""Build stratified diar manifest: RU / mix (synth_diar_ru) + EN (LibriConvo).

Only clips with MOSS timestamps+speakers. Mix = Cyrillic + Latin brands/code-switch.

  uv run python scripts/23_build_diar_lang_manifest.py \\
    --ru 16 --mix 11 --en 12 --out data/manifests/diar_langs_sample.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ru_bench.data import ClipRef, load_manifest, save_manifest
from ru_bench.hf_sources import load_libri_convo_en

_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]{2,}")


def lang_script_bucket(text: str) -> str:
    """ru | mix | en from script mix in transcript text."""
    nc = len(_CYR.findall(text))
    nl = len(_LAT.findall(text))
    if nc == 0 and nl > 0:
        return "en"
    if nl >= 2 and nc > 0:
        return "mix"
    return "ru"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev-manifest", type=Path, default=Path("data/manifests/dev_sample.json"))
    p.add_argument("--ru", type=int, default=16)
    p.add_argument("--mix", type=int, default=11)
    p.add_argument("--en", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("data/manifests/diar_langs_sample.json"))
    args = p.parse_args()

    diar = [c for c in load_manifest(args.dev_manifest) if c.domain == "synth_diar_ru"]
    buckets: dict[str, list[ClipRef]] = {"ru": [], "mix": [], "en": []}
    for c in diar:
        buckets[lang_script_bucket(c.target or c.text)].append(c)

    selected: list[ClipRef] = []
    selected.extend(buckets["ru"][: args.ru])
    selected.extend(buckets["mix"][: args.mix])

    if args.en > 0:
        en_clips = load_libri_convo_en(args.en)
        selected.extend(en_clips[: args.en])

    if not selected:
        raise SystemExit("no clips selected")

    save_manifest(selected, args.out)
    counts = {k: 0 for k in ("ru", "mix", "en")}
    for c in selected:
        b = "en" if c.domain == "libri_convo_en" else lang_script_bucket(c.target or c.text)
        counts[b] = counts.get(b, 0) + 1
    print(
        f"wrote {args.out} n={len(selected)} "
        f"ru={counts['ru']} mix={counts['mix']} en={counts['en']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

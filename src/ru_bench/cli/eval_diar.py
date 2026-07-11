"""GPU quality on diar clips with timestamps+speakers (RU / mix / EN).

Uses PyTorch + LoRA on CUDA (not ONNX). Metrics: CER/WER + cpCER/Δcp by lang.

  uv run python scripts/24_eval_gpu_diar_langs.py \\
    --manifest data/manifests/diar_langs_sample.json \\
    --checkpoint-dir checkpoints/lora_ru_fmt3 \\
    --out logs/eval_gpu_diar_langs.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import jiwer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from moss_transcribe_diarize import parse_transcript

from ru_bench.data import load_manifest
from ru_bench.metrics_moss import reference_segments, score_transcript_pair
from ru_bench.model_runner import load_model, run_one
from ru_bench.token_budget import max_new_tokens_for_duration

_CYR = re.compile(r"[\u0400-\u04FF]")
_LAT = re.compile(r"[A-Za-z]{2,}")


def lang_script_bucket(domain: str, text: str) -> str:
    if domain == "libri_convo_en" or domain.startswith("fleurs_en") or domain.startswith(
        "librispeech"
    ):
        return "en"
    nc = len(_CYR.findall(text))
    nl = len(_LAT.findall(text))
    if nc == 0 and nl > 0:
        return "en"
    if nl >= 2 and nc > 0:
        return "mix"
    return "ru"


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / max(len(vals), 1))


def _agg(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "cer": _mean([r["cer"] for r in rows]),
        "wer": _mean([r["wer"] for r in rows]),
        "cp_cer": _mean([r["cp_cer"] for r in rows]),
        "delta_cp": _mean([r["delta_cp"] for r in rows]),
        "rtf": _mean([r["rtf"] for r in rows]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/diar_langs_sample.json"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/lora_ru_fmt3"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-duration", type=float, default=35.0)
    p.add_argument("--out", type=Path, default=Path("logs/eval_gpu_diar_langs.json"))
    args = p.parse_args()

    clips = [c for c in load_manifest(args.manifest) if c.duration <= args.max_duration]
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit("no clips")

    print(
        f"Loading LoRA {args.checkpoint_dir} on CUDA (n={len(clips)})...",
        flush=True,
    )
    model, processor, device, dtype = load_model(adapter_path=str(args.checkpoint_dir))
    if device.type != "cuda":
        raise SystemExit(f"refusing non-CUDA device={device}")
    print(f"device={device} dtype={dtype}", flush=True)

    rows: list[dict] = []
    for i, clip in enumerate(clips):
        bucket = lang_script_bucket(clip.domain, clip.target or clip.text)
        max_new = max_new_tokens_for_duration(clip.duration)
        print(
            f"[{i + 1}/{len(clips)}] {clip.clip_id} {clip.domain} lang={bucket} "
            f"{clip.duration:.1f}s max_new={max_new}",
            flush=True,
        )
        t0 = time.perf_counter()
        result = run_one(
            model, processor, device, dtype, clip.audio_path, max_new
        )
        wall = time.perf_counter() - t0
        hyp = result["merged_text"]
        raw = result["raw_text"]
        cer = float(jiwer.cer(clip.text, hyp)) if clip.text.strip() else float("nan")
        wer = float(jiwer.wer(clip.text, hyp)) if clip.text.strip() else float("nan")
        hyp_segs = parse_transcript(raw) or []
        moss = score_transcript_pair(reference_segments(clip), hyp_segs)
        row = {
            "clip_id": clip.clip_id,
            "domain": clip.domain,
            "lang": bucket,
            "duration": clip.duration,
            "wall_s": wall,
            "rtf": wall / max(clip.duration, 1e-6),
            "cer": cer,
            "wer": wer,
            "cp_cer": moss["cp_cer"],
            "delta_cp": moss["delta_cp"],
            "merged": hyp,
            "raw": raw,
            "n_hyp_segs": len(hyp_segs),
            "n_hyp_spk": len({s.speaker for s in hyp_segs}),
        }
        rows.append(row)
        print(
            f"  cer={cer:.3f} wer={wer:.3f} cp={moss['cp_cer']:.3f} "
            f"dcp={moss['delta_cp']:.3f} rtf={row['rtf']:.2f} "
            f"spk={row['n_hyp_spk']} | {hyp[:80]}",
            flush=True,
        )

    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)

    summary = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint_dir),
        "device": str(device),
        "overall": _agg(rows),
        "by_lang": {lang: _agg(rs) for lang, rs in sorted(by_lang.items())},
        "per_clip": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("--- summary (GPU LoRA) ---", flush=True)
    o = summary["overall"]
    print(
        f"overall: n={o['n']} cer={o['cer']:.3f} wer={o['wer']:.3f} "
        f"cp={o['cp_cer']:.3f} dcp={o['delta_cp']:.3f} rtf={o['rtf']:.3f}",
        flush=True,
    )
    for lang, block in summary["by_lang"].items():
        print(
            f"{lang}: n={block['n']} cer={block['cer']:.3f} wer={block['wer']:.3f} "
            f"cp={block['cp_cer']:.3f} dcp={block['delta_cp']:.3f} rtf={block['rtf']:.3f}",
            flush=True,
        )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

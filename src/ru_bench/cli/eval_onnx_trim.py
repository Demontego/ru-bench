"""A/B quality: Whisper mel pad-3000 vs --trim-mel on a manifest (ONNX KV).

Loads sessions once; runs each clip twice (pad / trim); CER/WER + optional cpCER.
Aggregates overall + by lang_script_bucket (ru / mix / en) when clips have diar text.

  uv run python scripts/22_eval_onnx_trim_mel.py \\
    --manifest data/manifests/diar_langs_sample.json \\
    --limit 50 --max-duration 35 --quant cpu-fast --beam-size 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import jiwer
import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    prepare_inputs,
)

from ru_bench.data import load_manifest
from ru_bench.onnx_infer import (
    beam_decode_kv,
    decode_uses_input_ids,
    greedy_decode_kv,
    make_session,
    strip_gen_specials,
)
from ru_bench.metrics_moss import reference_segments, score_transcript_pair  # noqa: E402
from ru_bench.onnx_audio_dynpos import trim_mel_features  # noqa: E402
from ru_bench.token_budget import max_new_tokens_for_duration  # noqa: E402

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


def _pick_graphs(model_dir: Path, quant: str) -> tuple[Path, Path, Path, Path]:
    def graph(name: str, q: str) -> Path:
        if q == "fp32":
            return model_dir / f"{name}.onnx"
        p = model_dir / f"{name}.{q}.onnx"
        return p if p.is_file() else model_dir / f"{name}.onnx"

    if quant == "cpu-fast":
        audio = graph("audio", "fp32")
        prefill = graph("lm_prefill", "fp32")
        opt = model_dir / "lm_decode.opt.dynint8.onnx"
        decode = opt if opt.is_file() else graph("lm_decode", "dynint8")
    elif quant in {"opt", "fp32"}:
        audio = graph("audio", "fp32")
        prefill = graph("lm_prefill", "fp32")
        opt = model_dir / "lm_decode.opt.onnx"
        # Prefer optimized decode for CUDA/quality when present.
        if quant == "opt" or opt.is_file():
            decode = opt if opt.is_file() else graph("lm_decode", "fp32")
        else:
            decode = graph("lm_decode", "fp32")
    else:
        audio = graph("audio", quant)
        prefill = graph("lm_prefill", quant)
        decode = graph("lm_decode", quant)
    embed = model_dir / "embed.onnx"
    return audio, embed, prefill, decode


def _transcribe_one(
    *,
    sess_audio: ort.InferenceSession,
    sess_embed: ort.InferenceSession,
    sess_prefill: ort.InferenceSession,
    sess_decode: ort.InferenceSession,
    processor,
    audio_token_id: int,
    audio_path: Path,
    trim_mel: bool,
    mel_margin: int,
    beam_size: int,
    max_new: int,
) -> tuple[str, str, float]:
    messages = build_transcription_messages(str(audio_path))
    batch = prepare_inputs(processor, messages, device=None)
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    input_features = batch["input_features"].numpy().astype(np.float32)
    afl = batch["audio_feature_lengths"].numpy().astype(np.int64)
    if trim_mel:
        input_features = trim_mel_features(
            input_features, afl, margin_frames=mel_margin
        )
    prompt_len = int(input_ids.shape[1])
    eos_id = int(processor.tokenizer.eos_token_id)

    t0 = time.perf_counter()
    kw = dict(
        sess_audio=sess_audio,
        sess_embed=sess_embed,
        sess_prefill=sess_prefill,
        sess_decode=sess_decode,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        audio_feature_lengths=afl,
        audio_token_id=audio_token_id,
        eos_token_id=eos_id,
        max_new_tokens=max_new,
    )
    if beam_size > 1:
        out_ids = beam_decode_kv(**kw, beam_size=beam_size)
    else:
        out_ids = greedy_decode_kv(**kw)
    wall = time.perf_counter() - t0
    raw = strip_gen_specials(
        processor.tokenizer.decode(out_ids[0, prompt_len:], skip_special_tokens=False)
    )
    segs = parse_transcript(raw)
    merged = " ".join(s.text for s in sorted(segs, key=lambda s: s.start))
    if not merged.strip():
        # Flat Golos: parser may fail if no timestamps — use skip_special decode
        merged = processor.tokenizer.decode(
            out_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()
        raw = merged
    return raw, merged, wall


def _agg_rows(rows: list[dict], key: str, metric: str) -> float:
    vals = [r[key][metric] for r in rows if key in r and metric in r[key]]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _summary_block(rows: list[dict], modes: list[str] | None = None) -> dict:
    modes = modes or ["pad", "trim"]
    block: dict = {"n": len(rows)}
    present = [m for m in modes if any(m in r for r in rows)]
    for mode in present:
        block[mode] = {
            "cer": _agg_rows(rows, mode, "cer"),
            "wer": _agg_rows(rows, mode, "wer"),
            "cp_cer": _agg_rows(rows, mode, "cp_cer"),
            "delta_cp": _agg_rows(rows, mode, "delta_cp"),
            "rtf": _agg_rows(rows, mode, "rtf"),
        }
    if "pad" in block and "trim" in block:
        block["delta"] = {
            "cer": block["trim"]["cer"] - block["pad"]["cer"],
            "wer": block["trim"]["wer"] - block["pad"]["wer"],
            "cp_cer": block["trim"]["cp_cer"] - block["pad"]["cp_cer"],
            "delta_cp": block["trim"]["delta_cp"] - block["pad"]["delta_cp"],
            "rtf": block["trim"]["rtf"] - block["pad"]["rtf"],
        }
    return block


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/asr_sample.json"))
    p.add_argument("--model-dir", type=Path, default=Path("exports/moss_ru_fmt3_kv"))
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--max-duration", type=float, default=35.0)
    p.add_argument("--domain", action="append", default=None, help="Filter domain(s); repeatable")
    p.add_argument("--quant", default="fp32", help="Graph tag: fp32|cpu-fast|dynint8|...")
    p.add_argument("--beam-size", type=int, default=2)
    p.add_argument("--intra-op-threads", type=int, default=8)
    p.add_argument("--mel-margin-frames", type=int, default=0)
    p.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ORT providers (default: CUDA+CPU if CUDA EP available, else CPU)",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=["pad"],
        choices=["pad", "trim"],
        help="Which mel modes to run (default: pad only for quality)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("logs/eval_onnx_diar_langs.json"),
    )
    args = p.parse_args()

    providers = args.providers
    if providers is None:
        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    # Only hide CUDA when explicitly CPU-only (CPU ORT package path).
    if providers == ["CPUExecutionProvider"]:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    clips = load_manifest(args.manifest)
    clips = [c for c in clips if c.duration <= args.max_duration]
    if args.domain:
        allow = set(args.domain)
        clips = [c for c in clips if c.domain in allow]
    clips = clips[: args.limit]
    if not clips:
        raise SystemExit("no clips after filters")

    model_dir = args.model_dir
    meta = json.loads((model_dir / "kv_meta.json").read_text(encoding="utf-8"))
    audio_token_id = int(meta["audio_token_id"])
    audio_fp32, embed_path, prefill_path, decode_path = _pick_graphs(model_dir, args.quant)
    dynpos = model_dir / "audio.dynpos.onnx"
    need_trim = "trim" in args.modes
    if need_trim and not dynpos.is_file():
        raise SystemExit(
            f"missing {dynpos} — run: uv run python scripts/20_patch_audio_dynpos.py "
            f"--src {audio_fp32}"
        )

    sess_kw = {"intra_op_threads": args.intra_op_threads, "inter_op_threads": 1}
    sess_audio_pad = None
    sess_audio_trim = None
    if "pad" in args.modes:
        sess_audio_pad = make_session(audio_fp32, providers, **sess_kw)
    if need_trim:
        sess_audio_trim = make_session(dynpos, providers, **sess_kw)
    sess_embed = make_session(embed_path, providers, **sess_kw)
    sess_prefill = make_session(prefill_path, providers, **sess_kw)
    sess_decode = make_session(decode_path, providers, **sess_kw)
    print(
        f"providers={providers} quant={args.quant} modes={args.modes} "
        f"decode={decode_path.name} fused={decode_uses_input_ids(sess_decode)} "
        f"n={len(clips)} beam={args.beam_size}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(
        str(model_dir / "processor"), trust_remote_code=True
    )

    rows: list[dict] = []
    for i, clip in enumerate(clips):
        max_new = max_new_tokens_for_duration(clip.duration)
        bucket = lang_script_bucket(clip.domain, clip.target or clip.text)
        print(
            f"[{i + 1}/{len(clips)}] {clip.clip_id} {clip.domain} lang={bucket} "
            f"{clip.duration:.2f}s max_new={max_new}",
            flush=True,
        )
        pair: dict = {
            "clip_id": clip.clip_id,
            "domain": clip.domain,
            "lang": bucket,
            "duration": clip.duration,
        }
        mode_runs: list[tuple[str, ort.InferenceSession, bool]] = []
        if "pad" in args.modes and sess_audio_pad is not None:
            mode_runs.append(("pad", sess_audio_pad, False))
        if "trim" in args.modes and sess_audio_trim is not None:
            mode_runs.append(("trim", sess_audio_trim, True))
        for mode, sess_audio, trim in mode_runs:
            raw, merged, wall = _transcribe_one(
                sess_audio=sess_audio,
                sess_embed=sess_embed,
                sess_prefill=sess_prefill,
                sess_decode=sess_decode,
                processor=processor,
                audio_token_id=audio_token_id,
                audio_path=Path(clip.audio_path),
                trim_mel=trim,
                mel_margin=args.mel_margin_frames,
                beam_size=args.beam_size,
                max_new=max_new,
            )
            cer = float(jiwer.cer(clip.text, merged)) if clip.text.strip() else float("nan")
            wer = float(jiwer.wer(clip.text, merged)) if clip.text.strip() else float("nan")
            hyp_segs = parse_transcript(raw) or []
            moss = score_transcript_pair(reference_segments(clip), hyp_segs)
            if not hyp_segs:
                from moss_transcribe_diarize.transcript_parser import TranscriptSegment

                hyp_segs = [
                    TranscriptSegment(0.0, float(clip.duration), "S01", merged)
                ]
                moss = score_transcript_pair(reference_segments(clip), hyp_segs)

            pair[mode] = {
                "merged": merged,
                "raw": raw,
                "wall_s": wall,
                "rtf": wall / max(clip.duration, 1e-6),
                "cer": cer,
                "wer": wer,
                "cp_cer": moss["cp_cer"],
                "delta_cp": moss["delta_cp"],
            }
            print(
                f"  {mode}: cer={cer:.3f} wer={wer:.3f} cp={moss['cp_cer']:.3f} "
                f"dcp={moss['delta_cp']:.3f} rtf={pair[mode]['rtf']:.2f} "
                f"| {merged[:80]}",
                flush=True,
            )
        rows.append(pair)

    by_lang: dict[str, list[dict]] = {}
    for r in rows:
        by_lang.setdefault(r["lang"], []).append(r)

    overall = _summary_block(rows, modes=args.modes)
    primary = "pad" if "pad" in overall else args.modes[0]
    summary = {
        "manifest": str(args.manifest),
        "model_dir": str(args.model_dir),
        "quant": args.quant,
        "beam_size": args.beam_size,
        "providers": providers,
        "modes": args.modes,
        "n": overall["n"],
        "primary_mode": primary,
        "overall": overall,
        "by_lang": {
            lang: _summary_block(rs, modes=args.modes) for lang, rs in sorted(by_lang.items())
        },
        "per_clip": rows,
    }
    if primary in overall:
        summary[primary] = overall[primary]
    if "pad" in overall:
        summary["pad"] = overall["pad"]
    if "trim" in overall:
        summary["trim"] = overall["trim"]
    if "delta" in overall:
        summary["delta"] = overall["delta"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("--- summary ---", flush=True)
    def _fmt_mode(label: str, block: dict) -> None:
        if primary not in block:
            return
        m = block[primary]
        line = (
            f"{label}: n={block['n']} {primary} cer={m['cer']:.3f} wer={m['wer']:.3f} "
            f"cp={m['cp_cer']:.3f} dcp={m['delta_cp']:.3f} rtf={m['rtf']:.3f}"
        )
        if "delta" in block:
            d = block["delta"]
            line += (
                f" | Δcer={d['cer']:+.3f} Δcp={d['cp_cer']:+.3f} Δrtf={d['rtf']:+.3f}"
            )
        print(line, flush=True)

    _fmt_mode("overall", overall)
    for lang, block in summary["by_lang"].items():
        _fmt_mode(lang, block)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

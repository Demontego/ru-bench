"""ONNX KV-cache ASR inference CLI (chunk overlap, beam, cpu-fast).

  uv run ru-bench onnx infer --model-dir exports/moss_ru_fmt3_kv --audio clip.wav \\
    --quant cpu-fast --providers CPUExecutionProvider \\
    --chunk-sec 10 --overlap-sec 2 --beam-size 2 --no-trim-mel
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    prepare_inputs,
)
from transformers import AutoProcessor

from ru_bench.chunk_overlap import (
    merge_overlapping_chunks,
    offset_segments,
    plan_windows,
    segments_to_moss,
    write_wav_slice,
)
from ru_bench.onnx_audio_dynpos import trim_mel_features
from ru_bench.onnx_infer import (
    beam_decode_kv,
    cpu_only,
    decode_step_iobinding,
    decode_uses_input_ids,
    greedy_decode_kv,
    make_session,
    numpy_to_ortvalue,
    resolve_kv_paths,
    scatter_audio_np,
    strip_gen_specials,
)
from ru_bench.onnx_kv import N_LAYERS
from ru_bench.token_budget import max_new_tokens_for_chunk, max_new_tokens_for_duration


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=Path("exports/moss_ru_fmt3_kv"))
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Hard cap. Default: from --chunk-sec or audio duration",
    )
    p.add_argument(
        "--chunk-sec",
        type=float,
        default=None,
        help="Sliding window length (seconds). Preset budget: ≤12s→128, ≤35s→384",
    )
    p.add_argument(
        "--overlap-sec",
        type=float,
        default=None,
        help="Window overlap (default 2.0 when chunking; center-cut merge + speaker remap)",
    )
    p.add_argument("--beam-size", type=int, default=1, help="1=greedy; >1 beam search")
    p.add_argument("--providers", nargs="+", default=None)
    p.add_argument(
        "--providers-decode",
        nargs="+",
        default=None,
        help="Override providers for lm_decode only",
    )
    p.add_argument("--intra-op-threads", type=int, default=None)
    p.add_argument("--inter-op-threads", type=int, default=1)
    p.add_argument(
        "--tune-threads",
        action="store_true",
        help="Benchmark intra_op in {4,6,8,12,cpu} on first 40 decode steps",
    )
    p.add_argument(
        "--quant",
        choices=("fp32", "q8", "q4", "dynint8", "cpu-fast"),
        default="fp32",
        help="fp32 | q8/q4 MatMulNBits | dynint8 | cpu-fast (dynint8 decode)",
    )
    p.add_argument(
        "--trim-mel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trim Whisper mel pad via audio.dynpos.onnx (faster; small text drift)",
    )
    p.add_argument("--mel-margin-frames", type=int, default=0)
    p.add_argument("--audio-onnx", type=Path, default=None)
    args = p.parse_args()

    model_dir = args.model_dir
    meta = json.loads((model_dir / "kv_meta.json").read_text(encoding="utf-8"))
    audio_token_id = int(meta["audio_token_id"])

    providers = args.providers
    if providers is None and cpu_only(["CPUExecutionProvider"]):
        providers = ["CPUExecutionProvider"]
    providers_decode = args.providers_decode if args.providers_decode else providers
    has_ov = any(
        (p[0] if isinstance(p, (tuple, list)) else str(p)) == "OpenVINOExecutionProvider"
        for p in (providers or [])
    )
    if has_ov and args.providers_decode is None:
        providers = ["CPUExecutionProvider"]
        providers_decode = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        print(
            "note: OpenVINO → decode only; audio/embed/prefill on CPU "
            "(dynamic mel unsupported)",
            flush=True,
        )
    print(
        f"providers: audio/prefill={providers or 'default'} "
        f"decode={providers_decode or 'default'} quant={args.quant} ort={ort.__version__}",
        flush=True,
    )
    if "gpu" in ort.__file__.replace("\\", "/").lower() or "CUDAExecutionProvider" in (
        ort.get_available_providers()
    ):
        if cpu_only(providers):
            print(
                "note: onnxruntime-gpu build on CPU EP — for best CPU RTF use "
                "`uv sync --extra onnx-cpu`",
                flush=True,
            )

    audio_path, prefill_path, decode_path, embed_path = resolve_kv_paths(
        model_dir,
        quant=args.quant,
        trim_mel=bool(args.trim_mel),
        audio_onnx=args.audio_onnx,
    )
    if args.trim_mel:
        print(f"trim-mel=on audio={audio_path.name}", flush=True)

    intra = args.intra_op_threads
    sess_kw = {"intra_op_threads": intra, "inter_op_threads": args.inter_op_threads}
    sess_audio = make_session(audio_path, providers, **sess_kw)
    sess_embed = make_session(embed_path, providers, **sess_kw)
    sess_prefill = make_session(prefill_path, providers, **sess_kw)
    sess_decode = make_session(decode_path, providers_decode, **sess_kw)
    print(
        f"loaded {audio_path.name} / embed.onnx / "
        f"{prefill_path.name} / {decode_path.name}",
        flush=True,
    )

    processor_dir = model_dir / "processor"
    processor = AutoProcessor.from_pretrained(
        processor_dir if processor_dir.exists() else "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        trust_remote_code=True,
    )
    eos_id = int(processor.tokenizer.eos_token_id)

    samples, sr = sf.read(str(args.audio), always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    audio_sec = float(len(samples) / sr)

    chunk_sec = args.chunk_sec
    do_windows = chunk_sec is not None and audio_sec > float(chunk_sec) + 1e-6
    if do_windows:
        overlap_sec = float(args.overlap_sec) if args.overlap_sec is not None else 2.0
        windows = plan_windows(audio_sec, float(chunk_sec), overlap_sec)
    else:
        overlap_sec = float(args.overlap_sec or 0.0)
        windows = plan_windows(audio_sec, max(audio_sec, 0.01), 0.0)

    if args.max_new_tokens is not None:
        max_new = int(args.max_new_tokens)
        budget_src = "cli"
    elif chunk_sec is not None:
        max_new = max_new_tokens_for_chunk(chunk_sec)
        budget_src = f"chunk={chunk_sec:g}s"
    elif audio_sec > 0:
        max_new = max_new_tokens_for_duration(audio_sec)
        budget_src = f"duration={audio_sec:.1f}s"
    else:
        max_new = max_new_tokens_for_chunk(30)
        budget_src = "fallback-30s"

    hop = (float(chunk_sec) - overlap_sec) if do_windows else audio_sec
    print(
        f"max_new_tokens={max_new} ({budget_src}) windows={len(windows)} "
        f"chunk={chunk_sec} overlap={overlap_sec:g} hop={hop:g}",
        flush=True,
    )
    for i, w in enumerate(windows):
        print(f"  win[{i}] {w.start_sec:.2f}-{w.end_sec:.2f} ({w.duration:.2f}s)", flush=True)

    trim_mel = bool(args.trim_mel)

    def _prep_wav(wav_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        messages = build_transcription_messages(str(wav_path))
        batch = prepare_inputs(processor, messages, device=None)
        ids = batch["input_ids"].numpy()
        mask = batch["attention_mask"].numpy()
        ife = batch["input_features"].numpy().astype(np.float32)
        afl = batch["audio_feature_lengths"].numpy().astype(np.int64)
        if trim_mel:
            ife = trim_mel_features(ife, afl, margin_frames=int(args.mel_margin_frames))
        return ids, mask, ife, afl, int(ids.shape[1])

    tmpdir = Path(tempfile.mkdtemp(prefix="ru_bench_chunks_"))
    first_wav = tmpdir / "chunk_000.wav"
    write_wav_slice(samples, int(sr), windows[0], first_wav)
    input_ids, attention_mask, input_features, audio_feature_lengths, prompt_len = _prep_wav(
        first_wav
    )
    if trim_mel:
        print(
            f"trim-mel → {input_features.shape[-1]} "
            f"(afl={int(audio_feature_lengths.max())} margin={args.mel_margin_frames})",
            flush=True,
        )

    if args.tune_threads:
        cpus = os.cpu_count() or 8
        candidates = sorted({4, 6, 8, 12, min(16, cpus), cpus})
        candidates = [c for c in candidates if c >= 1]
        best_rate, best_intra = -1.0, (intra or min(8, cpus))
        audio_embeds = sess_audio.run(
            ["audio_embeds"],
            {
                "input_features": input_features,
                "audio_feature_lengths": audio_feature_lengths,
            },
        )[0]
        tok_embeds = sess_embed.run(
            ["inputs_embeds"], {"input_ids": input_ids.astype(np.int64)}
        )[0]
        pref_embeds = scatter_audio_np(
            tok_embeds.astype(np.float32),
            input_ids.astype(np.int64),
            audio_embeds.astype(np.float32),
            audio_token_id,
        )
        pref_outs = sess_prefill.run(
            None,
            {
                "inputs_embeds": pref_embeds,
                "attention_mask": attention_mask.astype(np.int64),
            },
        )
        base_past = [numpy_to_ortvalue(pref_outs[i + 1]) for i in range(N_LAYERS * 2)]
        first_id = int(np.argmax(pref_outs[0][0, -1]))
        fused = decode_uses_input_ids(sess_decode)
        tune_steps = 40
        for nthr in candidates:
            sdec = make_session(decode_path, providers_decode, intra_op_threads=nthr)
            past = list(base_past)
            mask = np.concatenate(
                [attention_mask.astype(np.int64), np.ones((1, 1), dtype=np.int64)],
                axis=1,
            )
            nid = first_id
            t1 = time.perf_counter()
            for _ in range(tune_steps):
                tin = (
                    np.array([[nid]], dtype=np.int64)
                    if fused
                    else sess_embed.run(
                        ["inputs_embeds"],
                        {"input_ids": np.array([[nid]], dtype=np.int64)},
                    )[0].astype(np.float32)
                )
                logits, past = decode_step_iobinding(
                    sdec,
                    fused_input_ids=fused,
                    token_input=tin,
                    attention_mask=mask,
                    past_ovs=past,
                )
                nid = int(np.argmax(logits[0, -1]))
                mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)
            rate = tune_steps / max(time.perf_counter() - t1, 1e-6)
            print(f"  intra_op={nthr} -> {rate:.2f} tok/s", flush=True)
            if rate > best_rate:
                best_rate, best_intra = rate, nthr
            del sdec
        print(f"tune pick intra_op={best_intra} ({best_rate:.2f} tok/s)", flush=True)
        sess_decode = make_session(decode_path, providers_decode, intra_op_threads=best_intra)
        sess_audio = make_session(audio_path, providers, intra_op_threads=best_intra)
        sess_embed = make_session(embed_path, providers, intra_op_threads=best_intra)
        sess_prefill = make_session(prefill_path, providers, intra_op_threads=best_intra)

    decode_fn = beam_decode_kv if args.beam_size > 1 else greedy_decode_kv
    chunk_results: list = []
    t_wall0 = time.perf_counter()

    for i, win in enumerate(windows):
        wav_i = tmpdir / f"chunk_{i:03d}.wav"
        if i == 0:
            wav_i = first_wav
            ids, mask, ife, afl, plen = (
                input_ids,
                attention_mask,
                input_features,
                audio_feature_lengths,
                prompt_len,
            )
        else:
            write_wav_slice(samples, int(sr), win, wav_i)
            ids, mask, ife, afl, plen = _prep_wav(wav_i)

        decode_kw: dict = dict(
            sess_audio=sess_audio,
            sess_embed=sess_embed,
            sess_prefill=sess_prefill,
            sess_decode=sess_decode,
            input_ids=ids,
            attention_mask=mask,
            input_features=ife,
            audio_feature_lengths=afl,
            audio_token_id=audio_token_id,
            eos_token_id=eos_id,
            max_new_tokens=max_new,
        )
        if args.beam_size > 1:
            decode_kw["beam_size"] = args.beam_size
        print(
            f"decode win[{i}] {win.start_sec:.2f}-{win.end_sec:.2f} "
            f"prompt_len={plen} mel_t={ife.shape[-1]}",
            flush=True,
        )
        out_ids = decode_fn(**decode_kw)
        gen = out_ids[0, plen:]
        raw_i = strip_gen_specials(
            processor.tokenizer.decode(gen, skip_special_tokens=False)
        )
        segs_local = parse_transcript(raw_i)
        segs_global = offset_segments(segs_local, win.start_sec)
        chunk_results.append((win, segs_global))
        print(
            f"  → segs={len(segs_local)} speakers="
            f"{sorted({s.speaker for s in segs_local})}",
            flush=True,
        )
        print(f"  raw: {raw_i[:200]}{'…' if len(raw_i) > 200 else ''}", flush=True)

    wall = time.perf_counter() - t_wall0
    segments = merge_overlapping_chunks(chunk_results, overlap_sec=overlap_sec)
    text = segments_to_moss(segments)
    merged = " ".join(s.text for s in sorted(segments, key=lambda s: s.start))
    if audio_sec <= 0 and segments:
        audio_sec = float(max(s.end for s in segments))
    rtf = wall / audio_sec if audio_sec > 0 else float("nan")
    print(
        f"RTF={rtf:.3f} (wall={wall:.1f}s / audio={audio_sec:.1f}s) "
        f"windows={len(windows)} "
        f"{'OK realtime' if rtf <= 1.0 else 'SLOWER than audio'}",
        flush=True,
    )
    print("--- raw (merged) ---", flush=True)
    print(text, flush=True)
    print("--- merged text ---", flush=True)
    print(merged, flush=True)
    for s in segments:
        print(f"[{s.start:.2f}-{s.end:.2f}][{s.speaker}] {s.text}", flush=True)
    print(f"segments={len(segments)} speakers={len({s.speaker for s in segments})}", flush=True)


if __name__ == "__main__":
    main()

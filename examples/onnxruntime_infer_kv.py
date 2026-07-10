"""Greedy ASR decode with split ONNX (audio once + embed + LM KV-cache).

Optimizations for CPU RTF:
  - fused ``lm_decode`` (``input_ids`` → embed inside graph)
  - IOBinding: KV stays as OrtValue (no 56× numpy copy/step)
  - tunable ``intra_op_num_threads``
  - skip CUDA preload when providers are CPU-only

Usage:
  uv sync --extra onnx-cpu   # preferred for CPU (not onnxruntime-gpu)
  uv run python scripts/14_export_onnx_kv.py --checkpoint-dir checkpoints/lora_ru_fmt3
  uv run python examples/onnxruntime_infer_kv.py \\
    --model-dir exports/moss_ru_fmt3_kv --audio Звонок.wav \\
    --quant cpu-fast --providers CPUExecutionProvider
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    prepare_inputs,
)

from ru_bench.onnx_kv import N_LAYERS, past_names, present_names
from ru_bench.token_budget import max_new_tokens_for_chunk, max_new_tokens_for_duration


def _cpu_only(providers: list[str] | None) -> bool:
    if not providers:
        return False
    return all(p == "CPUExecutionProvider" or p.startswith("CPU") for p in providers)


def make_session(
    onnx_path: Path,
    providers: list[str] | None,
    *,
    intra_op_threads: int | None = None,
    inter_op_threads: int = 1,
) -> ort.InferenceSession:
    if not _cpu_only(providers):
        try:
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.is_dir():
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except Exception as exc:  # noqa: BLE001
            print(f"torch lib PATH skipped: {exc}", flush=True)
        try:
            ort.preload_dlls(cuda=True, cudnn=True)
        except Exception as exc:  # noqa: BLE001
            print(f"preload_dlls skipped: {exc}", flush=True)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    n_threads = intra_op_threads if intra_op_threads is not None else min(8, os.cpu_count() or 4)
    so.intra_op_num_threads = max(1, n_threads)
    so.inter_op_num_threads = max(1, inter_op_threads)
    try:
        from onnxruntime_extensions import get_library_path

        so.register_custom_ops_library(get_library_path())
    except Exception as exc:  # noqa: BLE001
        print(f"extensions not loaded: {exc}", flush=True)

    providers = providers or ort.get_available_providers()
    print(
        f"session {onnx_path.name}: providers={providers} "
        f"intra_op={so.intra_op_num_threads} inter_op={so.inter_op_num_threads} "
        f"ort={ort.__version__}",
        flush=True,
    )
    return ort.InferenceSession(str(onnx_path), so, providers=providers)


def scatter_audio_np(
    token_embeds: np.ndarray,
    input_ids: np.ndarray,
    audio_embeds: np.ndarray,
    audio_token_id: int,
) -> np.ndarray:
    out = token_embeds.copy()
    flat_audio = audio_embeds.reshape(-1, audio_embeds.shape[-1])
    mask = input_ids.reshape(-1) == audio_token_id
    n = int(mask.sum())
    if n != flat_audio.shape[0]:
        raise ValueError(
            f"audio token count {n} != audio_embeds rows {flat_audio.shape[0]}"
        )
    out.reshape(-1, out.shape[-1])[mask] = flat_audio
    return out


def _decode_uses_input_ids(session: ort.InferenceSession) -> bool:
    return any(i.name == "input_ids" for i in session.get_inputs())


def _numpy_to_ortvalue(arr: np.ndarray) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(arr)


def decode_step_iobinding(
    session: ort.InferenceSession,
    *,
    fused_input_ids: bool,
    token_input: np.ndarray,
    attention_mask: np.ndarray,
    past_ovs: list[ort.OrtValue],
) -> tuple[np.ndarray, list[ort.OrtValue]]:
    """One decode step; ``past_ovs`` / present stay as OrtValue on CPU."""
    io = session.io_binding()
    if fused_input_ids:
        io.bind_cpu_input("input_ids", token_input)
    else:
        io.bind_cpu_input("inputs_embeds", token_input)
    io.bind_cpu_input("attention_mask", attention_mask)
    for name, ov in zip(past_names(), past_ovs, strict=True):
        io.bind_ortvalue_input(name, ov)
    io.bind_output("logits", "cpu")
    for name in present_names():
        io.bind_output(name, "cpu")
    session.run_with_iobinding(io)
    outs = io.get_outputs()
    logits = outs[0].numpy()
    return logits, list(outs[1:])


def greedy_decode_kv(
    *,
    sess_audio: ort.InferenceSession,
    sess_embed: ort.InferenceSession,
    sess_prefill: ort.InferenceSession,
    sess_decode: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    input_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    audio_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    log_every: int = 10,
) -> np.ndarray:
    t0 = time.perf_counter()
    fused = _decode_uses_input_ids(sess_decode)
    print(f"decode mode={'fused-input_ids' if fused else 'embeds+IOBinding'}", flush=True)

    audio_embeds = sess_audio.run(
        ["audio_embeds"],
        {
            "input_features": input_features.astype(np.float32),
            "audio_feature_lengths": audio_feature_lengths.astype(np.int64),
        },
    )[0]
    print(
        f"audio embeds {audio_embeds.shape} in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    tok_embeds = sess_embed.run(
        ["inputs_embeds"], {"input_ids": input_ids.astype(np.int64)}
    )[0]
    pref_embeds = scatter_audio_np(
        tok_embeds.astype(np.float32),
        input_ids.astype(np.int64),
        audio_embeds.astype(np.float32),
        audio_token_id,
    )

    t_pref = time.perf_counter()
    outs = sess_prefill.run(
        None,
        {
            "inputs_embeds": pref_embeds,
            "attention_mask": attention_mask.astype(np.int64),
        },
    )
    logits = outs[0]
    past_ovs = [_numpy_to_ortvalue(outs[i + 1]) for i in range(N_LAYERS * 2)]
    print(f"prefill done in {time.perf_counter() - t_pref:.2f}s", flush=True)

    ids = input_ids.copy()
    mask = attention_mask.astype(np.int64).copy()
    next_id = int(np.argmax(logits[0, -1]))
    ids = np.concatenate([ids, np.array([[next_id]], dtype=ids.dtype)], axis=1)
    mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)

    t_dec = time.perf_counter()
    for step in range(1, max_new_tokens):
        if next_id == eos_token_id:
            print(f"decode EOS at step={step}", flush=True)
            break

        if fused:
            token_in = np.array([[next_id]], dtype=np.int64)
        else:
            token_in = sess_embed.run(
                ["inputs_embeds"],
                {"input_ids": np.array([[next_id]], dtype=np.int64)},
            )[0].astype(np.float32)

        logits, past_ovs = decode_step_iobinding(
            sess_decode,
            fused_input_ids=fused,
            token_input=token_in,
            attention_mask=mask,
            past_ovs=past_ovs,
        )
        next_id = int(np.argmax(logits[0, -1]))
        ids = np.concatenate([ids, np.array([[next_id]], dtype=ids.dtype)], axis=1)
        mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)

        if log_every > 0 and step % log_every == 0:
            elapsed = time.perf_counter() - t_dec
            print(
                f"decode step={step} tok/s={step / max(elapsed, 1e-6):.2f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)
    return ids


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
        help="Preset budget: <=12s → 128 tok, <=35s → 384 tok (else duration formula)",
    )
    p.add_argument("--providers", nargs="+", default=None)
    p.add_argument(
        "--intra-op-threads",
        type=int,
        default=None,
        help="ORT intra_op_num_threads (default: min(8, cpu_count))",
    )
    p.add_argument(
        "--inter-op-threads",
        type=int,
        default=1,
        help="ORT inter_op_num_threads (default: 1)",
    )
    p.add_argument(
        "--tune-threads",
        action="store_true",
        help="Try intra_op in {4,6,8,12,cpu} on first 40 decode steps; pick fastest",
    )
    p.add_argument(
        "--quant",
        choices=("fp32", "q8", "q4", "dynint8", "cpu-fast"),
        default="fp32",
        help=(
            "fp32 | q8/q4 MatMulNBits | dynint8 (all) | "
            "cpu-fast (FP32 audio+prefill, dynint8 decode — best CPU RTF/quality)"
        ),
    )
    args = p.parse_args()

    model_dir = args.model_dir
    meta = json.loads((model_dir / "kv_meta.json").read_text(encoding="utf-8"))
    audio_token_id = int(meta["audio_token_id"])

    def graph(name: str, quant: str | None = None) -> Path:
        q = args.quant if quant is None else quant
        if q == "fp32":
            return model_dir / f"{name}.onnx"
        path = model_dir / f"{name}.{q}.onnx"
        if path.is_file():
            return path
        print(f"missing {path}, fallback {name}.onnx", flush=True)
        return model_dir / f"{name}.onnx"

    providers = args.providers
    if providers is None and _cpu_only(["CPUExecutionProvider"]):
        providers = ["CPUExecutionProvider"]
    print(
        f"providers: {providers or 'default'} quant={args.quant} ort={ort.__version__}",
        flush=True,
    )
    if "gpu" in ort.__file__.replace("\\", "/").lower() or "CUDAExecutionProvider" in (
        ort.get_available_providers()
    ):
        if _cpu_only(providers):
            print(
                "note: onnxruntime-gpu build on CPU EP — for best CPU RTF use "
                "`uv sync --extra onnx-cpu` (package onnxruntime, not onnxruntime-gpu)",
                flush=True,
            )

    if args.quant == "cpu-fast":
        audio_path = graph("audio", "fp32")
        prefill_path = graph("lm_prefill", "fp32")
        decode_path = graph("lm_decode", "dynint8")
    else:
        audio_path = graph("audio")
        prefill_path = graph("lm_prefill")
        decode_path = graph("lm_decode")

    intra = args.intra_op_threads
    if args.tune_threads:
        # Full tune needs loaded audio; do after processor prep with real decode.
        pass

    sess_kw = {
        "intra_op_threads": intra,
        "inter_op_threads": args.inter_op_threads,
    }
    sess_audio = make_session(audio_path, providers, **sess_kw)
    sess_embed = make_session(model_dir / "embed.onnx", providers, **sess_kw)
    sess_prefill = make_session(prefill_path, providers, **sess_kw)
    sess_decode = make_session(decode_path, providers, **sess_kw)
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

    messages = build_transcription_messages(str(args.audio))
    batch = prepare_inputs(processor, messages, device=None)
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    input_features = batch["input_features"].numpy().astype(np.float32)
    audio_feature_lengths = batch["audio_feature_lengths"].numpy().astype(np.int64)
    prompt_len = input_ids.shape[1]

    try:
        import soundfile as sf

        audio_sec = float(sf.info(str(args.audio)).duration)
    except Exception:  # noqa: BLE001
        audio_sec = 0.0

    if args.max_new_tokens is not None:
        max_new = int(args.max_new_tokens)
        budget_src = "cli"
    elif args.chunk_sec is not None:
        max_new = max_new_tokens_for_chunk(args.chunk_sec)
        budget_src = f"chunk={args.chunk_sec:g}s"
    elif audio_sec > 0:
        max_new = max_new_tokens_for_duration(audio_sec)
        budget_src = f"duration={audio_sec:.1f}s"
    else:
        max_new = max_new_tokens_for_chunk(30)
        budget_src = "fallback-30s"
    print(f"max_new_tokens={max_new} ({budget_src})", flush=True)

    if args.tune_threads:
        cpus = os.cpu_count() or 8
        candidates = sorted({4, 6, 8, 12, min(16, cpus), cpus})
        candidates = [c for c in candidates if c >= 1]
        best_rate, best_intra = -1.0, (intra or min(8, cpus))
        # Warmup shared audio/prefill once
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
        base_past = [_numpy_to_ortvalue(pref_outs[i + 1]) for i in range(N_LAYERS * 2)]
        first_id = int(np.argmax(pref_outs[0][0, -1]))
        fused = _decode_uses_input_ids(sess_decode)
        tune_steps = 40
        for nthr in candidates:
            sdec = make_session(decode_path, providers, intra_op_threads=nthr)
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
        sess_decode = make_session(decode_path, providers, intra_op_threads=best_intra)
        # Also rebuild other sessions with same thread count for consistency
        sess_audio = make_session(audio_path, providers, intra_op_threads=best_intra)
        sess_embed = make_session(model_dir / "embed.onnx", providers, intra_op_threads=best_intra)
        sess_prefill = make_session(prefill_path, providers, intra_op_threads=best_intra)

    t_wall0 = time.perf_counter()
    out_ids = greedy_decode_kv(
        sess_audio=sess_audio,
        sess_embed=sess_embed,
        sess_prefill=sess_prefill,
        sess_decode=sess_decode,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        audio_feature_lengths=audio_feature_lengths,
        audio_token_id=audio_token_id,
        eos_token_id=eos_id,
        max_new_tokens=max_new,
    )
    wall = time.perf_counter() - t_wall0
    gen = out_ids[0, prompt_len:]
    text = processor.tokenizer.decode(gen, skip_special_tokens=False)
    segments = parse_transcript(text)
    merged = " ".join(s.text for s in sorted(segments, key=lambda s: s.start))
    if audio_sec <= 0 and segments:
        audio_sec = float(max(s.end for s in segments))
    rtf = wall / audio_sec if audio_sec > 0 else float("nan")
    print(
        f"RTF={rtf:.3f} (wall={wall:.1f}s / audio={audio_sec:.1f}s) "
        f"{'OK realtime' if rtf <= 1.0 else 'SLOWER than audio'}",
        flush=True,
    )
    print("--- raw ---", flush=True)
    print(text, flush=True)
    print("--- merged ---", flush=True)
    print(merged, flush=True)
    for s in segments:
        print(f"[{s.start:.2f}-{s.end:.2f}][{s.speaker}] {s.text}", flush=True)


if __name__ == "__main__":
    main()

"""Greedy ASR decode with ONNX Runtime + onnxruntime-extensions.

- Main graph: one-step MOSS logits (prefer ``*.opt.onnx`` from transformers optimizer)
- Custom ops: ``SessionOptions.register_custom_ops_library(get_library_path())``
- Tokenizer: ``onnxruntime_extensions.pp_api.Tokenizer`` (Qwen2-compatible)
- Mel + full chat messages: still HF processor (audio feature extractor)

Docs:
  https://onnxruntime.ai/docs/performance/transformers-optimization.html
  https://onnxruntime.ai/docs/extensions/

Usage:
  uv sync --extra onnx
  uv run python scripts/12_export_onnx.py --checkpoint-dir checkpoints/lora_ru_e2
  uv run python examples/onnxruntime_infer.py --audio path/to.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    prepare_inputs,
)


def make_session(onnx_path: Path, providers: list[str] | None) -> ort.InferenceSession:
    """Load ORT session with onnxruntime-extensions custom ops registered."""
    # Reuse torch's CUDA/cuDNN DLLs (onnxruntime-gpu[cuda] extras are broken stubs).
    import os

    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        if torch_lib.is_dir():
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:  # noqa: BLE001
        print(f"torch lib PATH skipped: {exc}")

    try:
        ort.preload_dlls(cuda=True, cudnn=True)
    except Exception as exc:  # noqa: BLE001
        print(f"preload_dlls skipped: {exc}")

    so = ort.SessionOptions()
    try:
        from onnxruntime_extensions import get_library_path

        so.register_custom_ops_library(get_library_path())
        print(f"extensions: {get_library_path()}")
    except Exception as exc:  # noqa: BLE001 — demo should still run without Extensions
        print(f"extensions not loaded: {exc}")

    providers = providers or ort.get_available_providers()
    print(f"providers: {providers}")
    return ort.InferenceSession(str(onnx_path), so, providers=providers)


def demo_extensions_tokenizer(processor_dir: Path, text: str) -> list[int] | None:
    """Tokenize with Extensions pp_api (HfJsonTokenizer cannot parse Qwen2 JSON)."""
    try:
        from onnxruntime_extensions.pp_api import Tokenizer
    except ImportError:
        return None
    tok = Tokenizer(str(Path(processor_dir).resolve()))
    return list(tok.tokenize([text])[0])


def greedy_decode_ort(
    session: ort.InferenceSession,
    *,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    input_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    max_new_tokens: int,
    eos_token_id: int,
    log_every: int = 10,
) -> np.ndarray:
    """Append tokens via ORT logits until EOS or max_new_tokens."""
    import time

    ids = input_ids.copy()
    mask = attention_mask.copy()
    t0 = time.perf_counter()
    print(f"decode start max_new_tokens={max_new_tokens} prompt_len={ids.shape[1]}", flush=True)
    for step in range(max_new_tokens):
        logits = session.run(
            ["logits"],
            {
                "input_ids": ids,
                "attention_mask": mask,
                "input_features": input_features,
                "audio_feature_lengths": audio_feature_lengths,
            },
        )[0]
        next_id = int(np.argmax(logits[0, -1]))
        ids = np.concatenate([ids, np.array([[next_id]], dtype=ids.dtype)], axis=1)
        mask = np.concatenate([mask, np.ones((1, 1), dtype=mask.dtype)], axis=1)
        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            tok_s = (step + 1) / max(elapsed, 1e-6)
            print(
                f"decode step={step + 1}/{max_new_tokens} "
                f"tok/s={tok_s:.2f} elapsed={elapsed:.0f}s",
                flush=True,
            )
        if next_id == eos_token_id:
            print(f"decode EOS at step={step + 1}", flush=True)
            break
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--onnx",
        type=Path,
        default=Path("exports/moss_ru_e2_step.opt.onnx"),
        help="Prefer optimized graph; falls back to non-opt if missing",
    )
    p.add_argument(
        "--processor-dir",
        type=Path,
        default=None,
        help="Default: <onnx_dir>/processor or HF model id",
    )
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="ORT providers, e.g. CUDAExecutionProvider CPUExecutionProvider",
    )
    args = p.parse_args()

    onnx_path = args.onnx
    if not onnx_path.exists():
        fallback = onnx_path.with_name(onnx_path.name.replace(".opt.onnx", ".onnx"))
        if fallback.exists():
            print(f"missing {onnx_path}, using {fallback}")
            onnx_path = fallback
        else:
            raise FileNotFoundError(onnx_path)

    session = make_session(onnx_path, args.providers)
    print("inputs:", [(i.name, i.shape) for i in session.get_inputs()])

    processor_dir: Path | str
    if args.processor_dir is not None:
        processor_dir = args.processor_dir
    else:
        local = onnx_path.parent / "processor"
        processor_dir = local if local.exists() else "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    processor = AutoProcessor.from_pretrained(processor_dir, trust_remote_code=True)
    eos_id = int(processor.tokenizer.eos_token_id)

    if isinstance(processor_dir, Path) and processor_dir.exists():
        demo_ids = demo_extensions_tokenizer(processor_dir, "привет мир")
        if demo_ids is not None:
            print(f"extensions pp_api.Tokenizer demo ids={demo_ids[:12]}")

    messages = build_transcription_messages(str(args.audio))
    batch = prepare_inputs(processor, messages, device=None)
    input_ids = batch["input_ids"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    input_features = batch["input_features"].numpy().astype(np.float32)
    audio_feature_lengths = batch["audio_feature_lengths"].numpy().astype(np.int64)
    prompt_len = input_ids.shape[1]

    out_ids = greedy_decode_ort(
        session,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        audio_feature_lengths=audio_feature_lengths,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_id,
    )
    gen = out_ids[0, prompt_len:]
    text = processor.tokenizer.decode(gen, skip_special_tokens=False)
    segments = parse_transcript(text)
    merged = " ".join(s.text for s in sorted(segments, key=lambda s: s.start))

    print("--- raw ---")
    print(text)
    print("--- merged ---")
    print(merged)
    for s in segments:
        print(f"[{s.start:.2f}-{s.end:.2f}][{s.speaker}] {s.text}")


if __name__ == "__main__":
    main()

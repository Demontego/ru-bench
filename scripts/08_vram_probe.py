"""Binary-search max per-device microbatch that fits RTX 3090 (or current GPU).

Uses the same LoRA recipe as 06_finetune.py. Runs a few fwd+bwd steps per candidate.

  uv run python scripts/08_vram_probe.py --manifest data/manifests/asr_sample.json
"""

from __future__ import annotations

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ru_bench import config
from ru_bench.data import load_manifest
from ru_bench.lora_setup import build_lora_model
from ru_bench.model_runner import MODEL_ID
from ru_bench.train_loop import build_microbatch, resolve_train_device


def _vram_mb() -> tuple[float, float]:
    alloc = torch.cuda.memory_allocated() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    return alloc, peak


def _try_batch(model, processor, clips, device, dtype, batch_size: int, trials: int = 2) -> tuple[bool, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    try:
        for i in range(trials):
            micro = [clips[j % len(clips)] for j in range(i * batch_size, i * batch_size + batch_size)]
            batch = build_microbatch(micro, processor, device, dtype)
            with torch.autocast("cuda", dtype=dtype):
                out = model(**batch)
                loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        _, peak = _vram_mb()
        # Peak can exceed physical VRAM briefly via CUDA caching; treat over budget as fail.
        if peak > config.TRAIN_VRAM_SAFE_MIB:
            return False, peak
        return True, peak
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(config.ASR_MANIFEST_PATH))
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--target-eff", type=int, default=16, help="desired effective batch")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = resolve_train_device(args.device)
    if device.type != "cuda":
        raise SystemExit("CUDA required for VRAM probe")
    dtype = torch.bfloat16

    clips = load_manifest(args.manifest)
    # Prefer longer clips for conservative VRAM estimate.
    clips = sorted(clips, key=lambda c: c.duration, reverse=True)[: max(16, args.max_batch * 2)]
    print(f"Probe clips: {len(clips)}, longest={clips[0].duration:.2f}s")

    print("Loading model...")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
    model = build_lora_model(base)
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lo, hi = args.min_batch, args.max_batch
    best = 0
    best_peak = 0.0
    results: list[tuple[int, bool, float]] = []

    # Ascending probe then binary refine.
    candidates = []
    b = args.min_batch
    while b <= args.max_batch:
        candidates.append(b)
        b *= 2
    if args.max_batch not in candidates:
        candidates.append(args.max_batch)

    for batch_size in candidates:
        ok, peak = _try_batch(model, processor, clips, device, dtype, batch_size)
        results.append((batch_size, ok, peak))
        status = f"OK peak={peak:.0f}MiB" if ok else "OOM"
        print(f"  batch_size={batch_size}: {status}")
        if ok:
            best = batch_size
            best_peak = peak
        else:
            break

    # Binary search between best and first OOM.
    if best < args.max_batch:
        lo = best + 1
        hi = min(args.max_batch, best * 2 if best else args.max_batch)
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid in {b for b, _, _ in results}:
                # already tried
                if any(b == mid and ok for b, ok, _ in results):
                    lo = mid + 1
                else:
                    hi = mid - 1
                continue
            ok, peak = _try_batch(model, processor, clips, device, dtype, mid)
            results.append((mid, ok, peak))
            status = f"OK peak={peak:.0f}MiB" if ok else "OOM"
            print(f"  batch_size={mid}: {status}")
            if ok:
                best = mid
                best_peak = peak
                lo = mid + 1
            else:
                hi = mid - 1

    if best < 1:
        raise SystemExit("Even batch_size=1 OOMs — check other GPU users / model load")

    # Pick accum so eff ≈ target_eff (prefer multiples that land in 8–32).
    accum = max(1, round(args.target_eff / best))
    eff = best * accum
    # Prefer eff in [8, 32]
    if eff < 8:
        accum = max(1, (8 + best - 1) // best)
        eff = best * accum
    if eff > 32:
        accum = max(1, 32 // best)
        eff = best * accum

    print()
    print("=== RTX 3090 recommendation ===")
    print(f"max safe batch_size (microbatch): {best}  (peak ~{best_peak:.0f} MiB)")
    print(f"recommended: --batch-size {best} --batch-accum {accum}  (eff={eff})")
    print(f"config defaults to update: TRAIN_BATCH_SIZE={best} TRAIN_BATCH_ACCUM={accum}")

    # Persist recommendation into a small sidecar for humans / CI.
    out = config.DATA_DIR / "manifests" / "vram_probe_3090.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"batch_size={best}\nbatch_accum={accum}\neff={eff}\npeak_mib={best_peak:.0f}\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

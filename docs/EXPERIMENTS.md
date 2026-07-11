# Experiments log

Summary of tried paths — what worked, what rejected, why. Production path: `ru-bench onnx export-kv` + `ru-bench onnx infer`.

## LoRA training

| Recipe | Detail |
|--------|--------|
| Targets | Full `vq_adaptor`; Whisper LoRA r=32 (attn+FFN); Qwen LoRA r=8 (attn only) |
| Loss | Token-weighted CE: speakers ×4, timestamps ×2, text ×1; flat Golos ×0.4 |
| Data | RU-heavy train + EN flat/diar for bilingual retention (`05`, `25_inject_en_diar`) |
| RTX 3090 | `batch=4 accum=4` eff=16 → ~14GB peak (`08_vram_probe`) |
| Eval | Golos test `07`; retention `09 --mode en\|mix`; diar quality `24` |

**Open:** Golos crowd CDN shards may lack per-shard manifest — use farfield + HF golos10h.

## ONNX export (production)

**Path:** merge LoRA → `14_export_onnx_kv.py` → `exports/*/`

| Graph | Role |
|-------|------|
| `audio.onnx` | Whisper + VQ once |
| `embed.onnx` | Token embeddings |
| `lm_prefill.onnx` | Full prompt → past KV |
| `lm_decode.onnx` | Fused embed + KV step |

**Optimizations applied:**
1. ORT transformers optimize on decode (`opt_level=0`; rotary fusion **off** — breaks beam batch>1)
2. Dynamic INT8 on optimized decode → `lm_decode.opt.dynint8.onnx` (`--quant cpu-fast`)

## Runtime recipe (quality + realtime)

```bash
uv run python examples/onnxruntime_infer_kv.py \
  --model-dir exports/moss_ru_fmt3_kv --audio Звонок.wav \
  --quant cpu-fast --providers CPUExecutionProvider \
  --intra-op-threads 8 --chunk-sec 10 --overlap-sec 2 --beam-size 2 --no-trim-mel
```

| Mode | RTF | Quality |
|------|-----|---------|
| Single window, beam=2, cpu-fast | **~0.93** | Best clean realtime |
| Chunk 10s / overlap 2s / beam 2 | ~1.0–1.3 | Diar + timestamps OK |
| trim-mel (`20_patch_audio_dynpos`) | ~0.75 | Small text/timestamp drift |

Details: local `logs/RESULTS_zvonok.md` (gitignored); summary above.

## Rejected / bench-only

| Experiment | Result | Verdict |
|------------|--------|---------|
| Monolithic one-step ONNX (no KV) | Slow CPU greedy | **Removed** — use KV split |
| OpenVINO EP | ~1.6× slower decode | **Reject** |
| GPTQ MatMulNBits decode | Works; slower than dynint8 on CPU | **Reject** for prod |
| GQA INT4 LM | RTF win; diar collapse | **Reject** for prod |
| GQA quant KV | Broken prefill logits | **Reject** |
| GenAI C++ beam | RTF~1.46; needs separate export pack | **Not wired** — archived |
| Weight-only Q4/Q8 one-step | Timestamp/speaker drift on generative ASR | Bench only |

## trim-mel A/B (Golos asr_sample n=40)

Script: `ru-bench eval onnx-trim`

| | CER | WER | RTF |
|--|-----|-----|-----|
| pad-3000 | **0.062** | **0.229** | 1.071 |
| trim-mel | 0.089 | 0.289 | **0.462** |

Use trim-mel only when speed > small quality loss.

## ORT version

PyPI `onnxruntime>=1.27.0` sufficient for CPU GQA quant KV research. One-off GitHub DLL swap: `19_install_ort_from_git.py` (when PyPI lags).

## C++ port target

KV graph: `audio` → embed/scatter → `lm_prefill` → `lm_decode` (batched beam). Reuse `chunk_overlap` merge logic. Not GQA-ASR yet.

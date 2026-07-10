# ru-bench

Benchmarks [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) — a joint ASR + speaker-diarization model — on Russian audio.

MOSS-Transcribe-Diarize officially supports English and Chinese only. This benchmark measures how it performs on Russian anyway (CER/WER), using the [Golos](https://github.com/salute-developers/golos) Russian speech corpus. Poor results are an expected, valid finding here, not a sign of a broken pipeline — see [Assumptions and caveats](#assumptions-and-caveats).

Diarization itself is **not** evaluated: Golos ships with no speaker identifiers (anonymized at collection time), so there's no ground truth to score speaker labels against. This is an ASR-only (CER/WER) benchmark.

## Requirements

- Python 3.12 (pinned in `.python-version` / `pyproject.toml` — required by MOSS's Transformers 5.x stack)
- [uv](https://docs.astral.sh/uv/)
- CPU is fine (that's what this was built and verified against); a CUDA GPU will run inference faster but isn't required — `model_runner.py` auto-detects the device and uses float32 on CPU / bfloat16 on CUDA.
- ~2GB disk for the Golos test-split archive plus extracted audio.

## Setup

```bash
uv sync
```

This installs `ru-bench` itself plus `moss-transcribe-diarize` (as a git dependency), `jiwer`, and `soundfile`.

Verify:

```bash
uv run python -c "import moss_transcribe_diarize, jiwer, soundfile; print('ok')"
uv run python --version   # expect 3.12.x
```

## Usage

Run the four stages in order. Each is independently rerunnable and skips work already done.

```bash
# 1. Download Golos's test split (crowd + farfield, ~1.3GB) and sample N clips (default 100)
uv run python scripts/01_download_data.py --asr-n 100

# 2. Run MOSS-Transcribe-Diarize over the sample (resumable: reruns skip clips already processed)
uv run python scripts/02_run_inference.py

# 3. Compute CER/WER against Golos reference transcripts
uv run python scripts/03_compute_metrics.py

# 4. Generate the final report
uv run python scripts/04_report.py
```

Output:
- `data/manifests/asr_sample.json` — the sampled clips (id, domain, audio path, reference text, duration)
- `results/asr/{clip_id}.json` — raw model output per clip (segments, merged transcript)
- `results/metrics_asr.json` — CER/WER, corpus-level and per-clip
- `results/report.md` — human-readable summary: metrics, caveats, and the worst-CER examples

`data/` and `results/` are gitignored (not checked in — audio and generated results, not source).

### Scaling the sample

`01_download_data.py --asr-n <N>` re-samples deterministically (fixed seed). Re-run steps 2–4 afterward; step 2 only runs inference on clips it hasn't seen yet, so scaling up doesn't redo prior work.

To limit how many *new* clips step 2 processes in one go (e.g. for a quick check after changing the sample), use `--limit`:

```bash
uv run python scripts/02_run_inference.py --limit 5
```

## LoRA fine-tuning (experimental)

Beyond benchmarking, `ru-bench` includes a LoRA fine-tuning pipeline to try to improve Russian
performance: full fine-tune of the tiny `vq_adaptor` (audio→text projector), LoRA on the Whisper
encoder (attention + FFN, r=32), and a lighter LoRA on the Qwen3 decoder (attention only, r=8,
skipping its FFN). See `src/ru_bench/lora_setup.py` for the exact target modules.

### Requirements

- A CUDA GPU. LoRA still requires a full forward+backward pass through the frozen base model
  every step, so it only reduces optimizer/gradient memory, not compute — on CPU a single
  example's backward pass takes several minutes, making real training impractical. (MPS is fast
  but was observed to produce NaN losses after a few steps in local testing — likely an Apple
  Metal backend numerical issue, not a bug in this code; CPU forward+backward were verified
  numerically clean. Use CPU only for tiny mechanics-validation runs, not real training.)
- `peft` (installed via `uv sync`, already in `pyproject.toml`).

### Train data (multi-source)

`05_download_train_data.py` builds `data/manifests/train_sample.json` /
`data/manifests/dev_sample.json` from:

| Source | How | Notes |
|--------|-----|--------|
| `golos_farfield` | Sber CDN audio (~15.4GB) + `train_crowd9.tar` (~8GB) for manifests | farfield tar has wavs only |
| `golos10h` | HF `bond005/sberdevices_golos_10h_crowd` | avoid empty `SberDevices/Golos` |
| `fleurs_ru` / `fleurs_en` | HF `google/fleurs` | EN = modest retention, not train 50/50 |
| `synth_diar_ru` | HF `ivkond/synthetic-speech-diarization-ru` | multi-speaker + timestamps (format retention) |
| `cv_ru` | HF Common Voice 17 | often empty/gated on Hub — soft-skip |
| `librispeech_clean` | HF optional EN | pass explicitly; not in default mix |

**Format retention:** plain Golos/FLEURS labels are flat
`[0.00][S01]text[dur]` — that alone makes LoRA forget mid-timestamps / `[S02]`.
`synth_diar_ru` stores real multi-turn MOSS targets on `ClipRef.target`. Rebuild
manifests with `--sources mixed` (includes `synth_diar_ru`) before retraining.

**Loss:** token-weighted CE only — speakers ×4, timestamps ×2, text ×1;
flat Golos wrap ×0.4. Mid-train generate metrics: **CER / cpCER / Δcp**.

**Split policy:** eval = **10%** of pool, prefer **~50/50 RU/EN** (fair bilingual
metrics). Train = **all remaining** — RU-heavy is intended (primary = Russian);
modest EN share (~10–30% if available) keeps English retention. No huge EN
download just to balance train. Caps: `config.SOURCE_LIMITS`.

```bash
# Mixed default (farfield + golos10h + FLEURS RU/EN). Farfield alone ~15GB disk.
uv run python scripts/05_download_train_data.py --sources mixed

# HF-only (no 15GB farfield) — good for smoke / first train
uv run python scripts/05_download_train_data.py --sources hf --n-train 2000 --n-dev 150

# Legacy Golos-only
uv run python scripts/05_download_train_data.py --domain farfield
```

Disk ballpark: farfield tar 15GB + extract; golos10h+FLEURS wavs a few GB depending on limits in `config.SOURCE_LIMITS`.

### RTX 3090 batch (24GB)

LoRA recipe: Whisper r=32 + Qwen r=8 + full `vq_adaptor`, bf16 autocast.
VRAM probe (`scripts/08_vram_probe.py`) on longest ~13s clips:

- `batch_size=4` → peak ~14GB — **safe**
- `batch_size=8` → peak ~26GB — **unsafe** on 24GB

Recommended: **`--batch-size 4 --batch-accum 4` → eff=16**.

```bash
# Re-probe if recipe/clip length changes
uv run python scripts/08_vram_probe.py --manifest data/manifests/asr_sample.json

# 2. Fine-tune (defaults: batch=4 accum=4 eff=16, 1000 steps, lr 1e-4)
uv run python scripts/06_finetune.py --device cuda --steps 1000

# 3. Evaluate the fine-tuned checkpoint on the same 100-clip test set used for the baseline
uv run python scripts/07_eval_finetuned.py --checkpoint-dir checkpoints/lora_ru
```

Output: `checkpoints/lora_ru/` (PEFT adapter + `vq_adaptor` weights, saved together via PEFT's
`modules_to_save`), `results/asr_finetuned/` and `results/metrics_asr_finetuned.json` (parallel to
the baseline results, so neither overwrites the other).

CUDA smoke (not full train):

```bash
uv run python scripts/06_finetune.py --train-manifest data/manifests/train_sample.json \
  --dev-manifest none --limit 16 --steps 4 --batch-size 4 --batch-accum 1 \
  --device cuda --checkpoint-dir checkpoints/smoke_3090
```

### Known open item

`train_data.py`'s `--domain crowd` / `--sources golos_crowd` path is unverified: Golos's crowd
train shards may only bundle the full manifest inside `train_crowd9.tar` rather than each shard
being self-contained like `farfield`/`test` are. The code raises a clear error if no manifest is
found. Prefer `golos_farfield` + HF `golos10h` instead of CDN crowd shards.

## ONNX export + onnxruntime

Exports a **one-step** graph (prompt+audio → last-token logits), then:

1. [ORT transformers optimizer](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
   (`model_type=gpt2`, Qwen heads=16 / hidden=1024, **`opt_level=0`**) → `*.opt.onnx`
2. [onnxruntime-extensions](https://onnxruntime.ai/docs/extensions/) —
   register custom ops + `pp_api.Tokenizer` for Qwen vocab
   (`HfJsonTokenizer` ONNX op **cannot** parse Qwen2 `tokenizer.json`)

Greedy decode loop in Python. No KV-cache — demo speed.

```bash
uv sync --extra onnx

# Merge LoRA + export + optimize + Extensions tokenizer check (~4GB weights)
uv run python scripts/12_export_onnx.py \
  --checkpoint-dir checkpoints/lora_ru_e2 \
  --output exports/moss_ru_e2_step.onnx

# Optional GPU FP16 optimize:
#   ... --float16 --use-gpu

# Weight-only MatMulNBits (Q4 / INT8) — size-first, GPU/CPU
uv run python scripts/13_quantize_onnx.py --method nbits --bits 4
uv run python scripts/13_quantize_onnx.py --method nbits --bits 8

# CPU dynamic INT8 (activations runtime-quantized; best for CPUExecutionProvider)
uv run python scripts/13_quantize_onnx.py --method dynamic --exclude-lm-head

# Split KV export (audio once + embed + lm_prefill + lm_decode) — preferred for CPU/GPU speed
uv run python scripts/14_export_onnx_kv.py \
  --checkpoint-dir checkpoints/lora_ru_fmt3 \
  --output-dir exports/moss_ru_fmt3_kv \
  --sample-audio Звонок.wav

# CPU runtime: use onnxruntime (not onnxruntime-gpu) for best CPU EP
#   uv remove onnxruntime-gpu --optional onnx
#   uv sync --extra onnx-cpu

uv run python examples/onnxruntime_infer_kv.py \
  --model-dir exports/moss_ru_fmt3_kv \
  --audio Звонок.wav \
  --quant cpu-fast \
  --providers CPUExecutionProvider \
  --tune-threads

# Token budgets (train+dev): ~10s → 128, ~30s → 384; or auto from duration
#   --chunk-sec 10 | --chunk-sec 30 | omit both for duration formula

# Infer (Extensions custom ops + prefer *.opt.onnx)
uv run python examples/onnxruntime_infer.py \
  --onnx exports/moss_ru_e2_step.opt.onnx \
  --audio data/raw/golos/test/farfield/files/<clip>.wav \
  --max-new-tokens 128
```

Artifacts:
- `exports/moss_ru_e2_step.onnx` (+ `.data`)
- `exports/moss_ru_e2_step.opt.onnx` (+ `.data`) — fused (`opt_level=0`)
- `exports/*.opt.q4.onnx` / `*.opt.q8.onnx` — weight-only MatMulNBits
- `exports/moss_ru_fmt3_kv/` — **preferred**: `audio` + `embed` + `lm_prefill` + `lm_decode` (KV-cache)
- `exports/extensions_tokenizer.json` — pp_api.Tokenizer metadata
- `exports/processor/` — HF processor (mel) + tokenizer for Extensions

Caveats:
- Mel still HF feature extractor; text tokenize demo via Extensions `pp_api`.
- `onnxruntime-extensions==0.15.0` pinned (0.15.2 has no Windows cp312 wheel).
- Single-chunk / batch=1.
- Monolithic one-step graph has no KV-cache (slow on CPU). Use `14_export_onnx_kv.py`.
- Default optimizer ``opt_level=0``. ``opt_level>=1`` invalidates hybrid Whisper+Qwen.
- Q4/INT8 is weight-only; A/B on real audio — generative ASR can degrade on timestamps/speakers.
- KV export duplicates LM weights in prefill+decode (~2× LM disk); audio/embed shared.

## Assumptions and caveats

- **Dataset**: [Golos](https://github.com/salute-developers/golos) (Sberdevices), test split, downloaded directly from Sberdevices' CDN. Distributed under a custom public license (attribution + share-alike, modeled on CC BY-SA 4.0 — see `license/en_us.pdf` in the Golos repo). Attribution: Karpov et al., ["Golos: Russian Dataset for Speech Research"](https://arxiv.org/abs/2106.10161), Interspeech 2021.
- **Out-of-domain language**: Russian is not an officially supported language for this model — expect (and this benchmark confirms) meaningfully degraded transcription quality compared to English/Chinese.
- **No diarization evaluation**: Golos has no speaker-ID metadata by design, so there's no ground truth to build a diarization/DER track from. Only ASR quality (CER/WER) is measured here.
- **Small sample, CPU-only**: default sample size is 100 clips out of Golos's ~11.8k-clip test set, run on CPU in float32 — sized for practicality, not statistical power. Increase `--asr-n` for a more representative sample if you have the compute budget.

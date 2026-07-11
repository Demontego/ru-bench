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
# 1. Download Golos test split and sample N clips (default 100)
uv run ru-bench bench download --asr-n 100

# 2. Run MOSS over the sample (resumable)
uv run ru-bench bench infer

# 3. Compute CER/WER
uv run ru-bench bench metrics

# 4. Generate report
uv run ru-bench bench report
```

Full command reference: [`docs/COMMANDS.md`](docs/COMMANDS.md). Repo layout: [`docs/LAYOUT.md`](docs/LAYOUT.md).

Output:
- `data/manifests/asr_sample.json` — the sampled clips (id, domain, audio path, reference text, duration)
- `results/asr/{clip_id}.json` — raw model output per clip (segments, merged transcript)
- `results/metrics_asr.json` — CER/WER, corpus-level and per-clip
- `results/report.md` — human-readable summary: metrics, caveats, and the worst-CER examples

`data/` and `results/` are gitignored (not checked in — audio and generated results, not source).

### Commands

One CLI: **`uv run ru-bench <group> <action>`** — see [`docs/COMMANDS.md`](docs/COMMANDS.md).

| Group | Actions |
|-------|---------|
| `bench` | download, infer, metrics, report |
| `data` | manifests, inject-en, diar-sample, error-mine |
| `train` | finetune, vram-probe |
| `eval` | golos, retention, diar, onnx-trim |
| `onnx` | export-kv, patch-audio, install-ort, infer |

Experiments log: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

### Scaling the sample

`01_download_data.py --asr-n <N>` re-samples deterministically (fixed seed). Re-run steps 2–4 afterward; step 2 only runs inference on clips it hasn't seen yet, so scaling up doesn't redo prior work.

To limit how many *new* clips step 2 processes in one go (e.g. for a quick check after changing the sample), use `--limit`:

```bash
uv run ru-bench bench infer --limit 5
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
| `fleurs_ru` / `fleurs_en` | HF `google/fleurs` | EN flat ASR retention |
| `synth_diar_ru` | HF `ivkond/synthetic-speech-diarization-ru` | RU multi-speaker + timestamps |
| `libri_convo_en` | HF `gedeonmate/LibriConvo-segmented` | EN multi-speaker + timestamps |
| `cv_ru` | HF Common Voice 17 | often empty/gated on Hub — soft-skip |
| `librispeech_clean` | HF LibriSpeech clean | EN flat ASR retention |

**Format retention:** plain Golos/FLEURS labels are flat
`[0.00][S01]text[dur]` — that alone makes LoRA forget mid-timestamps / `[S02]`.
`synth_diar_ru` + `libri_convo_en` store real multi-turn MOSS targets on
`ClipRef.target`. Rebuild manifests with `--sources mixed`, or inject EN diar
into existing manifests:

```bash
uv run ru-bench data inject-en --limit 1500
```

Train sampler: `TRAIN_EN_SAMPLE_RATIO` of slots are EN; within those,
`TRAIN_EN_DIAR_RATIO` prefer multi-spk EN over flat.

**Loss:** token-weighted CE only — speakers ×4, timestamps ×2, text ×1;
flat Golos wrap ×0.4; EN clips ×`LOSS_W_EN_CLIP`. Mid-train generate metrics:
**CER / cpCER / Δcp** (+ `en_cer` on flat/diar EN).

**Split policy:** eval = **10%** of pool, prefer **~50/50 RU/EN** (fair bilingual
metrics). Train = **all remaining** — RU-heavy is intended (primary = Russian);
EN flat + EN diar keep bilingual diarization. Caps: `config.SOURCE_LIMITS`.

```bash
# Mixed default (farfield + golos10h + FLEURS RU/EN). Farfield alone ~15GB disk.
uv run ru-bench data manifests --sources mixed

# HF-only (no 15GB farfield) — good for smoke / first train
uv run ru-bench data manifests --sources hf --n-train 2000 --n-dev 150

# Legacy Golos-only
uv run ru-bench data manifests --domain farfield
```

Disk ballpark: farfield tar 15GB + extract; golos10h+FLEURS wavs a few GB depending on limits in `config.SOURCE_LIMITS`.

### RTX 3090 batch (24GB)

LoRA recipe: Whisper r=32 + Qwen r=8 + full `vq_adaptor`, bf16 autocast.
VRAM probe (`ru-bench train vram-probe`) on longest ~13s clips:

- `batch_size=4` → peak ~14GB — **safe**
- `batch_size=8` → peak ~26GB — **unsafe** on 24GB

Recommended: **`--batch-size 4 --batch-accum 4` → eff=16**.

```bash
# Re-probe if recipe/clip length changes
uv run ru-bench train vram-probe --manifest data/manifests/asr_sample.json

# 2. Fine-tune (defaults: batch=4 accum=4 eff=16, 1000 steps, lr 1e-4)
uv run ru-bench train finetune --device cuda --steps 1000

# 3. Evaluate the fine-tuned checkpoint on the same 100-clip test set used for the baseline
uv run ru-bench eval golos --checkpoint-dir checkpoints/lora_ru
```

Output: `checkpoints/lora_ru/` (PEFT adapter + `vq_adaptor` weights, saved together via PEFT's
`modules_to_save`), `results/asr_finetuned/` and `results/metrics_asr_finetuned.json` (parallel to
the baseline results, so neither overwrites the other).

CUDA smoke (not full train):

```bash
uv run ru-bench train finetune --train-manifest data/manifests/train_sample.json \
  --dev-manifest none --limit 16 --steps 4 --batch-size 4 --batch-accum 1 \
  --device cuda --checkpoint-dir checkpoints/smoke_3090
```

### Known open item

`train_data.py`'s `--domain crowd` / `--sources golos_crowd` path is unverified: Golos's crowd
train shards may only bundle the full manifest inside `train_crowd9.tar` rather than each shard
being self-contained like `farfield`/`test` are. The code raises a clear error if no manifest is
found. Prefer `golos_farfield` + HF `golos10h` instead of CDN crowd shards.

## ONNX export + inference

Production path: split KV graphs + Python runtime. Experimental tries (GQA, GenAI, GPTQ, monolithic step) summarized in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

```bash
uv sync --extra onnx-cpu   # CPU EP — prefer over onnxruntime-gpu for CPU RTF

# Export (merge LoRA → audio + embed + lm_prefill + lm_decode)
uv run ru-bench onnx export-kv \
  --checkpoint-dir checkpoints/lora_ru_fmt3 \
  --output-dir exports/moss_ru_fmt3_kv \
  --sample-audio Звонок.wav \
  --optimize-lm

# Optional: trim-mel audio graph (speed knob — small quality drift)
uv run ru-bench onnx patch-audio --src exports/moss_ru_fmt3_kv/audio.onnx

# Infer (preferred realtime recipe)
uv run ru-bench onnx infer \
  --model-dir exports/moss_ru_fmt3_kv \
  --audio Звонок.wav \
  --quant cpu-fast \
  --providers CPUExecutionProvider \
  --intra-op-threads 8 \
  --chunk-sec 10 --overlap-sec 2 --beam-size 2 --no-trim-mel
```

**Artifacts:** `exports/moss_ru_fmt3_kv/` — `audio.onnx`, `embed.onnx`, `lm_prefill.onnx`, `lm_decode.onnx` (+ `*.opt.onnx`, `lm_decode.opt.dynint8.onnx` for cpu-fast), `processor/`, `kv_meta.json`.

**Caveats:** Mel via HF feature extractor; single batch; KV export duplicates LM weights in prefill+decode; `onnxruntime-extensions==0.15.0` pinned (no cp312 wheel for 0.15.2).

**Eval / A/B:** `ru-bench eval onnx-trim` (pad vs trim-mel). ORT 1.27+ from PyPI; one-off GitHub install: `ru-bench onnx install-ort`.

## Assumptions and caveats

- **Dataset**: [Golos](https://github.com/salute-developers/golos) (Sberdevices), test split, downloaded directly from Sberdevices' CDN. Distributed under a custom public license (attribution + share-alike, modeled on CC BY-SA 4.0 — see `license/en_us.pdf` in the Golos repo). Attribution: Karpov et al., ["Golos: Russian Dataset for Speech Research"](https://arxiv.org/abs/2106.10161), Interspeech 2021.
- **Out-of-domain language**: Russian is not an officially supported language for this model — expect (and this benchmark confirms) meaningfully degraded transcription quality compared to English/Chinese.
- **No diarization evaluation**: Golos has no speaker-ID metadata by design, so there's no ground truth to build a diarization/DER track from. Only ASR quality (CER/WER) is measured here.
- **Small sample, CPU-only**: default sample size is 100 clips out of Golos's ~11.8k-clip test set, run on CPU in float32 — sized for practicality, not statistical power. Increase `--asr-n` for a more representative sample if you have the compute budget.

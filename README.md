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

### Usage

```bash
# 1. Download Golos's train split (farfield domain, ~15.4GB, self-contained manifest)
uv run python scripts/05_download_train_data.py --domain farfield

# 2. Fine-tune (defaults: 1000 steps, batch-accum 8, lr 1e-4)
uv run python scripts/06_finetune.py --device cuda --steps 1000 --batch-accum 8

# 3. Evaluate the fine-tuned checkpoint on the same 100-clip test set used for the baseline,
#    and print both numbers side by side
uv run python scripts/07_eval_finetuned.py --checkpoint-dir checkpoints/lora_ru
```

Output: `checkpoints/lora_ru/` (PEFT adapter + `vq_adaptor` weights, saved together via PEFT's
`modules_to_save`), `results/asr_finetuned/` and `results/metrics_asr_finetuned.json` (parallel to
the baseline results, so neither overwrites the other).

For a quick mechanics-only smoke test (not real training — just confirms the pipeline runs),
point `--train-manifest`/`--dev-manifest` at the existing `data/manifests/asr_sample.json` and use
`--limit`/`--steps`/`--batch-accum` to keep it tiny, e.g.:

```bash
uv run python scripts/06_finetune.py --train-manifest data/manifests/asr_sample.json \
  --dev-manifest "" --limit 2 --steps 2 --batch-accum 1 --device cpu \
  --checkpoint-dir checkpoints/smoke_test
```

### Known open item

`train_data.py`'s `--domain crowd` path is unverified: Golos's crowd train shards may only bundle
the full manifest inside `train_crowd9.tar` rather than each shard being self-contained like
`farfield`/`test` are. The code raises a clear error (rather than silently doing the wrong thing)
if no manifest is found in a downloaded shard. `--domain farfield` (the default) sidesteps this
entirely.

## Assumptions and caveats

- **Dataset**: [Golos](https://github.com/salute-developers/golos) (Sberdevices), test split, downloaded directly from Sberdevices' CDN. Distributed under a custom public license (attribution + share-alike, modeled on CC BY-SA 4.0 — see `license/en_us.pdf` in the Golos repo). Attribution: Karpov et al., ["Golos: Russian Dataset for Speech Research"](https://arxiv.org/abs/2106.10161), Interspeech 2021.
- **Out-of-domain language**: Russian is not an officially supported language for this model — expect (and this benchmark confirms) meaningfully degraded transcription quality compared to English/Chinese.
- **No diarization evaluation**: Golos has no speaker-ID metadata by design, so there's no ground truth to build a diarization/DER track from. Only ASR quality (CER/WER) is measured here.
- **Small sample, CPU-only**: default sample size is 100 clips out of Golos's ~11.8k-clip test set, run on CPU in float32 — sized for practicality, not statistical power. Increase `--asr-n` for a more representative sample if you have the compute budget.

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

## Assumptions and caveats

- **Dataset**: [Golos](https://github.com/salute-developers/golos) (Sberdevices), test split, downloaded directly from Sberdevices' CDN. Distributed under a custom public license (attribution + share-alike, modeled on CC BY-SA 4.0 — see `license/en_us.pdf` in the Golos repo). Attribution: Karpov et al., ["Golos: Russian Dataset for Speech Research"](https://arxiv.org/abs/2106.10161), Interspeech 2021.
- **Out-of-domain language**: Russian is not an officially supported language for this model — expect (and this benchmark confirms) meaningfully degraded transcription quality compared to English/Chinese.
- **No diarization evaluation**: Golos has no speaker-ID metadata by design, so there's no ground truth to build a diarization/DER track from. Only ASR quality (CER/WER) is measured here.
- **Small sample, CPU-only**: default sample size is 100 clips out of Golos's ~11.8k-clip test set, run on CPU in float32 — sized for practicality, not statistical power. Increase `--asr-n` for a more representative sample if you have the compute budget.

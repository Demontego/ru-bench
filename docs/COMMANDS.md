# Commands

Single entry point: **`uv run ru-bench <group> <action> [options]`**

## Benchmark (Golos)

```bash
uv run ru-bench bench download --asr-n 100
uv run ru-bench bench infer
uv run ru-bench bench infer --limit 5
uv run ru-bench bench metrics
uv run ru-bench bench report
```

## Data manifests

```bash
uv run ru-bench data manifests --sources mixed
uv run ru-bench data manifests --sources hf --n-train 2000 --n-dev 150
uv run ru-bench data inject-en --limit 1500
uv run ru-bench data diar-sample --ru 16 --mix 11 --en 12
uv run ru-bench data error-mine --help
```

## Training

```bash
uv run ru-bench train vram-probe
uv run ru-bench train finetune --device cuda --steps 1000
uv run ru-bench train finetune --limit 16 --steps 4 --checkpoint-dir checkpoints/smoke
```

## Evaluation

```bash
uv run ru-bench eval golos --checkpoint-dir checkpoints/lora_ru
uv run ru-bench eval retention --mode en
uv run ru-bench eval retention --mode mix
uv run ru-bench eval diar --manifest data/manifests/diar_langs_sample.json
uv run ru-bench eval onnx-trim --manifest data/manifests/asr_sample.json --limit 50
```

## ONNX

```bash
uv sync --extra onnx-cpu

uv run ru-bench onnx export-kv \
  --checkpoint-dir checkpoints/lora_ru_fmt3 \
  --output-dir exports/moss_ru_fmt3_kv \
  --optimize-lm

uv run ru-bench onnx patch-audio --src exports/moss_ru_fmt3_kv/audio.onnx

uv run ru-bench onnx infer \
  --model-dir exports/moss_ru_fmt3_kv \
  --audio Звонок.wav \
  --quant cpu-fast \
  --providers CPUExecutionProvider \
  --intra-op-threads 8 \
  --chunk-sec 10 --overlap-sec 2 --beam-size 2 --no-trim-mel
```

Rejected experiments: [EXPERIMENTS.md](EXPERIMENTS.md).

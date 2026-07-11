# Layout

```
src/ru_bench/
  cli/              # all user-facing commands (thin argparse wrappers)
  config.py         # paths, limits, token budgets
  data.py           # ClipRef, Golos download/manifest I/O
  model_runner.py   # PyTorch load + run_one
  metrics_asr.py    # CER/WER
  metrics_moss.py   # cpCER / diar metrics
  report.py         # results/report.md
  train_*.py        # LoRA train loop, data, collate, loss
  onnx_*.py         # export, KV graphs, runtime decode, trim-mel patch
  chunk_overlap.py  # sliding-window merge
  hf_sources.py     # HF corpora loaders

docs/
  COMMANDS.md       # CLI reference (single entry: ru-bench)
  EXPERIMENTS.md    # what we tried / rejected

scripts/            # deprecated — see README
examples/           # deprecated — see README
```

**Entry point:** `uv run ru-bench <group> <action>`

No numbered scripts. Library code in `src/ru_bench/`, commands in `src/ru_bench/cli/`.

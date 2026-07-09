import json
from datetime import datetime, timezone

from ru_bench import config

ASSUMPTIONS = """\
## Assumptions and caveats

- **Model is used out of its officially supported languages.** MOSS-Transcribe-Diarize's own \
documentation lists English and Chinese; Russian is not advertised as supported. Poor scores \
below reflect that gap, not a broken pipeline.
- **Dataset**: [Golos](https://github.com/salute-developers/golos) (Sberdevices), test split \
(crowd + farfield domains), downloaded directly from Sberdevices' CDN. Golos is distributed \
under a custom public license (attribution + share-alike required, modeled on CC BY-SA 4.0; \
see `license/en_us.pdf` in the Golos repo) — attribution: Karpov et al., "Golos: Russian \
Dataset for Speech Research", Interspeech 2021.
- **CPU-only, small sample.** Inference ran on CPU in float32; the sample size below is a small \
subset of the full test set, not the full 11.8k clips, to keep runtime practical.
- **Diarization is not evaluated.** Golos deliberately ships with no speaker identifiers \
(anonymized at collection time), so a diarization/DER track could not be built from it. This \
report covers ASR quality (CER/WER) only.
"""


def worst_examples(per_clip: list[dict], k: int = 5) -> list[dict]:
    return sorted(per_clip, key=lambda c: c["cer"], reverse=True)[:k]


def build_report(metrics: dict, seed: int = config.SEED) -> str:
    lines = [
        "# ru-bench: MOSS-Transcribe-Diarize on Russian audio (Golos, ASR)",
        "",
        f"Run date: {datetime.now(timezone.utc).date().isoformat()}  ",
        f"Sample size: {metrics['n']} clips  ",
        f"Seed: {seed}  ",
        "",
        ASSUMPTIONS,
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| N | {metrics['n']} |",
        f"| CER | {metrics['cer']:.4f} |",
        f"| WER | {metrics['wer']:.4f} |",
        "",
        "## Worst examples (highest per-clip CER)",
        "",
    ]
    for ex in worst_examples(metrics["per_clip"]):
        lines.append(f"- clip `{ex['clip_id']}` (CER={ex['cer']:.2f})")
        lines.append(f"  - reference: {ex['reference']}")
        lines.append(f"  - hypothesis: {ex['hypothesis']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    with open(config.ASR_METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    report = build_report(metrics)
    config.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()

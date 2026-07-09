import json

import jiwer

from ru_bench import config
from ru_bench.data import ClipRef, load_manifest


def load_hypothesis(clip: ClipRef, results_dir=config.ASR_RESULTS_DIR) -> str:
    path = results_dir / f"{clip.clip_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["merged_text"]


def compute_asr_metrics(clips: list[ClipRef] | None = None, results_dir=config.ASR_RESULTS_DIR) -> dict:
    clips = clips if clips is not None else load_manifest()
    clips = [c for c in clips if (results_dir / f"{c.clip_id}.json").exists()]

    refs = [c.text for c in clips]
    hyps = [load_hypothesis(c, results_dir) for c in clips]

    per_clip = []
    for clip, ref, hyp in zip(clips, refs, hyps):
        per_clip.append(
            {
                "clip_id": clip.clip_id,
                "reference": ref,
                "hypothesis": hyp,
                "cer": jiwer.cer(ref, hyp),
                "wer": jiwer.wer(ref, hyp),
            }
        )

    cer = jiwer.process_characters(refs, hyps).cer if refs else float("nan")
    wer = jiwer.process_words(refs, hyps).wer if refs else float("nan")

    return {"n": len(clips), "cer": cer, "wer": wer, "per_clip": per_clip}


def save_metrics(metrics: dict, path=config.ASR_METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

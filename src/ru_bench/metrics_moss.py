"""MOSS-style objective metrics: CER, cpCER, Δcp.

Paper/repo (OpenMOSS/MOSS-Transcribe-Diarize, arXiv:2601.01554):
  - CER: speaker-independent character error (time-ordered concat)
  - cpCER: concatenated minimum-permutation CER (speaker streams)
  - Δcp = cpCER - CER  (speaker-attribution penalty)
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Sequence

import jiwer

from moss_transcribe_diarize.transcript_parser import TranscriptSegment, parse_transcript

from ru_bench.data import ClipRef


def _norm_chars(text: str) -> str:
    """Character-level normalize: drop whitespace, lowercase (RU/EN/ZH-friendly)."""
    return "".join(text.split()).lower()


def reference_segments(clip: ClipRef) -> list[TranscriptSegment]:
    if clip.target:
        segs = parse_transcript(clip.target)
        if segs:
            return segs
    return [
        TranscriptSegment(start=0.0, end=float(clip.duration), speaker="S01", text=clip.text)
    ]


def _concat_time_order(segments: Sequence[TranscriptSegment]) -> str:
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    return _norm_chars("".join(s.text for s in ordered))


def _concat_by_speaker(segments: Sequence[TranscriptSegment]) -> dict[str, str]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for seg in sorted(segments, key=lambda s: (s.start, s.end)):
        buckets[seg.speaker].append(seg.text)
    return {spk: _norm_chars("".join(parts)) for spk, parts in buckets.items()}


def cer_speaker_independent(
    ref: Sequence[TranscriptSegment],
    hyp: Sequence[TranscriptSegment],
) -> float:
    ref_t = _concat_time_order(ref)
    hyp_t = _concat_time_order(hyp)
    if not ref_t:
        return 0.0 if not hyp_t else 1.0
    return float(jiwer.cer(ref_t, hyp_t))


def cp_cer(
    ref: Sequence[TranscriptSegment],
    hyp: Sequence[TranscriptSegment],
    *,
    max_speakers: int = 6,
) -> float:
    """Concatenated minimum-permutation CER over speaker streams."""
    ref_by = _concat_by_speaker(ref)
    hyp_by = _concat_by_speaker(hyp)
    ref_keys = sorted(ref_by.keys())
    hyp_keys = sorted(hyp_by.keys())

    n = max(len(ref_keys), len(hyp_keys), 1)
    if n > max_speakers:
        # Too many perms — truncate to most frequent (longest) streams.
        ref_keys = sorted(ref_keys, key=lambda k: len(ref_by[k]), reverse=True)[:max_speakers]
        hyp_keys = sorted(hyp_keys, key=lambda k: len(hyp_by[k]), reverse=True)[:max_speakers]
        n = max(len(ref_keys), len(hyp_keys), 1)

    while len(ref_keys) < n:
        pad = f"__ref_pad_{len(ref_keys)}"
        ref_keys.append(pad)
        ref_by[pad] = ""
    while len(hyp_keys) < n:
        pad = f"__hyp_pad_{len(hyp_keys)}"
        hyp_keys.append(pad)
        hyp_by[pad] = ""

    ref_streams = [ref_by[k] for k in ref_keys]
    best = float("inf")
    for perm in itertools.permutations(hyp_keys):
        hyp_streams = [hyp_by[k] for k in perm]
        # Skip all-empty pairs noise: jiwer on empty refs.
        usable_ref: list[str] = []
        usable_hyp: list[str] = []
        for r, h in zip(ref_streams, hyp_streams, strict=True):
            if not r and not h:
                continue
            if not r:
                # Pure insertion stream — count as 100% error of hyp length via dummy.
                usable_ref.append(" ")
                usable_hyp.append(h or " ")
            else:
                usable_ref.append(r)
                usable_hyp.append(h)
        if not usable_ref:
            best = min(best, 0.0)
            continue
        score = float(jiwer.process_characters(usable_ref, usable_hyp).cer)
        if score < best:
            best = score
    return 0.0 if best is float("inf") else best


def delta_cp(cer: float, cpcer: float) -> float:
    return cpcer - cer


def score_transcript_pair(
    ref: Sequence[TranscriptSegment],
    hyp: Sequence[TranscriptSegment],
) -> dict[str, float]:
    cer = cer_speaker_independent(ref, hyp)
    cpcer = cp_cer(ref, hyp)
    return {"cer": cer, "cp_cer": cpcer, "delta_cp": delta_cp(cer, cpcer)}


def score_clip_hypothesis(clip: ClipRef, hyp_text: str) -> dict[str, float]:
    ref = reference_segments(clip)
    hyp = parse_transcript(hyp_text)
    if not hyp and hyp_text.strip():
        # Model dumped flat text without brackets — treat as single S01 stream.
        hyp = [
            TranscriptSegment(start=0.0, end=float(clip.duration), speaker="S01", text=hyp_text)
        ]
    return score_transcript_pair(ref, hyp)


def aggregate_clip_scores(scores: list[dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {"n": 0, "cer": float("nan"), "cp_cer": float("nan"), "delta_cp": float("nan")}
    n = len(scores)
    cer = sum(s["cer"] for s in scores) / n
    cpcer = sum(s["cp_cer"] for s in scores) / n
    return {"n": n, "cer": cer, "cp_cer": cpcer, "delta_cp": delta_cp(cer, cpcer)}

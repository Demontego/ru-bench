"""Sliding-window audio chunking with overlap merge for MOSS diar transcripts.

Windows: ``chunk_sec`` length, hop = ``chunk_sec - overlap_sec``.
Merge: center-cut in the overlap + speaker remap via turn overlap in the
shared region (local ``S01`` in chunk N may be global ``S02``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from moss_transcribe_diarize.transcript_parser import TranscriptSegment

from ru_bench.moss_format import format_moss_segment


@dataclass(frozen=True)
class TimeWindow:
    """Half-open ``[start_sec, end_sec)`` on the source timeline."""

    start_sec: float
    end_sec: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def plan_windows(
    duration_sec: float,
    chunk_sec: float,
    overlap_sec: float,
    *,
    min_tail_sec: float = 1.0,
) -> list[TimeWindow]:
    """Cover ``[0, duration]`` with overlapping windows.

    Last stub shorter than ``min_tail_sec`` is absorbed into the previous
    window (extend end) instead of spawning a tiny decode.
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    if chunk_sec <= 0:
        raise ValueError(f"chunk_sec must be > 0, got {chunk_sec}")
    if overlap_sec < 0:
        raise ValueError(f"overlap_sec must be >= 0, got {overlap_sec}")
    if overlap_sec >= chunk_sec:
        raise ValueError(
            f"overlap_sec ({overlap_sec}) must be < chunk_sec ({chunk_sec})"
        )

    if duration_sec <= chunk_sec:
        return [TimeWindow(0.0, duration_sec)]

    hop = chunk_sec - overlap_sec
    windows: list[TimeWindow] = []
    start = 0.0
    while start < duration_sec:
        end = min(start + chunk_sec, duration_sec)
        windows.append(TimeWindow(start, end))
        if end >= duration_sec - 1e-9:
            break
        start += hop

    if len(windows) >= 2 and windows[-1].duration < min_tail_sec:
        prev = windows[-2]
        windows[-2] = TimeWindow(prev.start_sec, duration_sec)
        windows.pop()
    return windows


def offset_segments(
    segments: list[TranscriptSegment], offset_sec: float
) -> list[TranscriptSegment]:
    """Shift chunk-local timestamps onto the global timeline."""
    return [
        TranscriptSegment(
            start=float(s.start) + offset_sec,
            end=float(s.end) + offset_sec,
            speaker=s.speaker,
            text=s.text,
        )
        for s in segments
    ]


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _clamp_segment(
    seg: TranscriptSegment, t0: float, t1: float
) -> TranscriptSegment | None:
    """Keep intersection of ``seg`` with ``[t0, t1]``; drop empty."""
    lo = max(float(seg.start), t0)
    hi = min(float(seg.end), t1)
    if hi <= lo + 1e-3:
        return None
    return TranscriptSegment(start=lo, end=hi, speaker=seg.speaker, text=seg.text)


def _speaker_overlap_matrix(
    left: list[TranscriptSegment],
    right: list[TranscriptSegment],
    t0: float,
    t1: float,
) -> dict[str, dict[str, float]]:
    """``right_spk -> left_spk -> overlap_sec`` inside ``[t0, t1]``."""
    out: dict[str, dict[str, float]] = {}
    for r in right:
        r0, r1 = max(float(r.start), t0), min(float(r.end), t1)
        if r1 <= r0:
            continue
        bucket = out.setdefault(r.speaker, {})
        for le in left:
            ov = _interval_overlap(r0, r1, float(le.start), float(le.end))
            if ov > 0:
                bucket[le.speaker] = bucket.get(le.speaker, 0.0) + ov
    return out


def _speaker_at(segments: list[TranscriptSegment], t: float) -> str | None:
    """Speaker covering time ``t`` (prefer latest start if ties)."""
    hit = [
        s
        for s in segments
        if float(s.start) - 1e-6 <= t <= float(s.end) + 1e-6
    ]
    if not hit:
        return None
    hit.sort(key=lambda s: float(s.start))
    return hit[-1].speaker


def remap_speakers(
    left: list[TranscriptSegment],
    right: list[TranscriptSegment],
    overlap_t0: float,
    overlap_t1: float,
    *,
    next_speaker_id: int,
    cut_t: float | None = None,
    leftover_pool: list[TranscriptSegment] | None = None,
) -> tuple[list[TranscriptSegment], dict[str, str], int]:
    """Map right-chunk local speakers onto left's global labels via overlap.

    Also seeds map from speakers active at ``cut_t`` (center-cut), so turns
    that only touch the boundary still align. ``leftover_pool`` (usually all
    merged-so-far turns) supplies globals for unpaired leftovers.
    """
    scores = _speaker_overlap_matrix(left, right, overlap_t0, overlap_t1)
    used_left: set[str] = set()
    mapping: dict[str, str] = {}

    if cut_t is not None:
        lspk = _speaker_at(left, cut_t - 1e-3)
        rspk = _speaker_at(right, cut_t + 1e-3)
        if lspk and rspk:
            mapping[rspk] = lspk
            used_left.add(lspk)

    for rspk, by_left in sorted(
        scores.items(), key=lambda kv: -max(kv[1].values()) if kv[1] else 0.0
    ):
        if rspk in mapping:
            continue
        best_left = None
        best_ov = 0.0
        for lspk, ov in by_left.items():
            if lspk in used_left:
                continue
            if ov > best_ov:
                best_ov, best_left = ov, lspk
        if best_left is not None and best_ov >= 0.15:
            mapping[rspk] = best_left
            used_left.add(best_left)

    pool = leftover_pool if leftover_pool is not None else left
    left_spks = {s.speaker for s in pool}
    right_spks = {s.speaker for s in right}
    unused_left = sorted(left_spks - used_left)
    unmapped_right = sorted(right_spks - set(mapping))
    while unused_left and unmapped_right:
        rspk = unmapped_right.pop(0)
        lspk = unused_left.pop(0)
        mapping[rspk] = lspk
        used_left.add(lspk)

    sid = next_speaker_id
    for r in right:
        if r.speaker in mapping:
            continue
        mapping[r.speaker] = f"S{sid:02d}"
        sid += 1

    remapped = [
        TranscriptSegment(
            start=float(s.start),
            end=float(s.end),
            speaker=mapping[s.speaker],
            text=s.text,
        )
        for s in right
    ]
    return remapped, mapping, sid


def _next_speaker_id(segments: list[TranscriptSegment]) -> int:
    ids: list[int] = []
    for s in segments:
        sp = s.speaker.strip().upper()
        if sp.startswith("S") and sp[1:].isdigit():
            ids.append(int(sp[1:]))
    return (max(ids) + 1) if ids else 1


def merge_overlapping_chunks(
    chunk_results: list[tuple[TimeWindow, list[TranscriptSegment]]],
    *,
    overlap_sec: float,
) -> list[TranscriptSegment]:
    """Center-cut merge. ``chunk_results`` segments must already be global-time.

    Keep ``[cut_lo, cut_hi]`` per window:
      cut_lo = start           (first) else start + overlap/2
      cut_hi = end             (last)  else end - overlap/2
    """
    if not chunk_results:
        return []
    if len(chunk_results) == 1:
        return list(chunk_results[0][1])

    half = max(0.0, float(overlap_sec) * 0.5)
    merged: list[TranscriptSegment] = []
    next_id = 1

    for i, (win, segs) in enumerate(chunk_results):
        is_first = i == 0
        is_last = i == len(chunk_results) - 1
        cut_lo = win.start_sec if is_first else win.start_sec + half
        cut_hi = win.end_sec if is_last else win.end_sec - half

        piece = [
            c
            for s in segs
            if (c := _clamp_segment(s, cut_lo, cut_hi)) is not None
        ]

        if is_first:
            merged.extend(piece)
            next_id = _next_speaker_id(merged)
            continue

        prev_win, prev_segs = chunk_results[i - 1]
        ov_t0 = win.start_sec
        ov_t1 = min(prev_win.end_sec, win.end_sec)
        # Score against previous chunk's full turns (not yet center-cut).
        remapped, _map, next_id = remap_speakers(
            prev_segs,
            segs,
            ov_t0,
            ov_t1,
            next_speaker_id=max(next_id, _next_speaker_id(merged)),
            cut_t=cut_lo,
            leftover_pool=merged,
        )
        piece = [
            c
            for s in remapped
            if (c := _clamp_segment(s, cut_lo, cut_hi)) is not None
        ]
        merged.extend(piece)

    merged.sort(key=lambda s: (s.start, s.end))
    merged = _stitch_adjacent(merged)
    return _fill_timeline_gaps(merged, chunk_results, overlap_sec=overlap_sec)


def _fill_timeline_gaps(
    merged: list[TranscriptSegment],
    chunk_results: list[tuple[TimeWindow, list[TranscriptSegment]]],
    *,
    overlap_sec: float,
    min_gap: float = 0.35,
) -> list[TranscriptSegment]:
    """If center-cut left a hole, pull covering turns from any chunk."""
    if not merged or not chunk_results:
        return merged
    t_end = max(w.end_sec for w, _ in chunk_results)
    out = list(merged)
    changed = True
    while changed:
        changed = False
        out.sort(key=lambda s: (s.start, s.end))
        cursor = 0.0
        inserts: list[TranscriptSegment] = []
        for s in out:
            if float(s.start) > cursor + min_gap:
                g0, g1 = cursor, float(s.start)
                cand = _best_gap_fill(chunk_results, g0, g1)
                if cand is not None:
                    inserts.append(cand)
                    changed = True
            cursor = max(cursor, float(s.end))
        if t_end > cursor + min_gap:
            cand = _best_gap_fill(chunk_results, cursor, t_end)
            if cand is not None:
                inserts.append(cand)
                changed = True
        if inserts:
            # Remap gap fills onto existing speaker set when possible.
            pool = out
            remapped_ins: list[TranscriptSegment] = []
            next_id = _next_speaker_id(out)
            for ins in inserts:
                remapped, _, next_id = remap_speakers(
                    pool,
                    [ins],
                    float(ins.start),
                    float(ins.end),
                    next_speaker_id=next_id,
                    leftover_pool=pool,
                )
                remapped_ins.extend(remapped)
            out.extend(remapped_ins)
    out.sort(key=lambda s: (s.start, s.end))
    return _stitch_adjacent(out)


def _best_gap_fill(
    chunk_results: list[tuple[TimeWindow, list[TranscriptSegment]]],
    g0: float,
    g1: float,
) -> TranscriptSegment | None:
    """Longest clamped turn covering most of ``[g0, g1]``."""
    best: TranscriptSegment | None = None
    best_ov = 0.0
    for _win, segs in chunk_results:
        for s in segs:
            c = _clamp_segment(s, g0, g1)
            if c is None:
                continue
            ov = float(c.end) - float(c.start)
            if ov > best_ov:
                best_ov, best = ov, c
    return best if best_ov >= 0.2 else None


def _norm_words(text: str) -> list[str]:
    return [w for w in text.lower().replace("-", " ").split() if w]


def _dedupe_join_text(left: str, right: str) -> str:
    """Concatenate without repeating overlapping word suffix/prefix."""
    a, b = left.strip(), right.strip()
    if not a:
        return b
    if not b:
        return a
    wa, wb = _norm_words(a), _norm_words(b)
    if not wa:
        return b
    if not wb:
        return a
    sa, sb = " ".join(wa), " ".join(wb)
    if sb in sa:
        return a
    if sa in sb:
        return b
    # Near-duplicate turns (overlap ASR): keep longer original.
    inter = len(set(wa) & set(wb))
    union = len(set(wa) | set(wb))
    if union and inter / union >= 0.55:
        return a if len(a) >= len(b) else b
    best = 0
    for k in range(1, min(len(wa), len(wb)) + 1):
        if wa[-k:] == wb[:k]:
            best = k
    if best == 0:
        return f"{a} {b}".strip()
    if best >= len(wb):
        return a
    # Rebuild from original tokens roughly: drop first ``best`` words of right.
    right_words = b.split()
    return f"{a} {' '.join(right_words[best:])}".strip()


def _stitch_adjacent(
    segments: list[TranscriptSegment], *, gap_sec: float = 0.15
) -> list[TranscriptSegment]:
    """Join consecutive same-speaker turns; dedupe overlap text."""
    if not segments:
        return []
    out: list[TranscriptSegment] = [segments[0]]
    for s in segments[1:]:
        prev = out[-1]
        if (
            s.speaker == prev.speaker
            and float(s.start) <= float(prev.end) + gap_sec
            and s.text
        ):
            text = _dedupe_join_text(prev.text, s.text)
            out[-1] = TranscriptSegment(
                start=prev.start,
                end=max(prev.end, s.end),
                speaker=prev.speaker,
                text=text,
            )
        else:
            out.append(s)
    return out


def segments_to_moss(segments: list[TranscriptSegment]) -> str:
    """Serialize to MOSS compact string."""
    return "".join(
        format_moss_segment(s.start, s.end, s.speaker, s.text)
        for s in sorted(segments, key=lambda x: x.start)
        if s.text.strip()
    )


def write_wav_slice(
    samples: np.ndarray,
    sr: int,
    window: TimeWindow,
    path: Path,
) -> Path:
    """Write ``samples[window]`` to ``path`` (mono float/int ok for soundfile)."""
    path = Path(path)
    i0 = int(round(window.start_sec * sr))
    i1 = int(round(window.end_sec * sr))
    i0 = max(0, min(i0, len(samples)))
    i1 = max(i0, min(i1, len(samples)))
    sf.write(str(path), samples[i0:i1], sr)
    return path

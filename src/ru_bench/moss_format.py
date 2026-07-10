"""Build MOSS compact transcript targets from segment lists."""

from __future__ import annotations

from typing import Any

from ru_bench.data import ClipRef


def format_moss_segment(start: float, end: float, speaker: str, text: str) -> str:
    """One turn: ``[start][Sxx]text[end]`` (MOSS compact format)."""
    text = text.strip()
    speaker = speaker.strip().upper()
    if not speaker.startswith("S"):
        speaker = f"S{int(speaker):02d}"
    elif len(speaker) == 2 and speaker[1].isdigit():
        speaker = f"S{int(speaker[1:]):02d}"
    return f"[{start:.2f}][{speaker}]{text}[{end:.2f}]"


def speakers_to_moss_target(speakers: list[dict[str, Any]], *, max_end: float | None = None) -> str:
    """Convert ivkond-style speaker turns into a MOSS target string.

    Drops empty text; optionally truncates turns past ``max_end`` (seconds).
    """
    parts: list[str] = []
    for seg in sorted(speakers, key=lambda s: float(s["start"])):
        start = float(seg["start"])
        end = float(seg["end"])
        if max_end is not None:
            if start >= max_end:
                break
            end = min(end, max_end)
        text = str(seg.get("text") or "").strip()
        if not text or end <= start:
            continue
        sid = int(seg["speaker_id"])
        parts.append(format_moss_segment(start, end, f"S{sid:02d}", text))
    return "".join(parts)


def target_text(clip: ClipRef) -> str:
    """Training label: prefer stored multi-segment target, else flat Golos wrap."""
    if clip.target:
        return clip.target
    return f"[0.00][S01]{clip.text}[{clip.duration:.2f}]"

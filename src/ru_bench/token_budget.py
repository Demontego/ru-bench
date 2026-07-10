"""Decode-length budgets for MOSS compact transcripts vs audio duration."""

from __future__ import annotations

from ru_bench import config


def max_new_tokens_for_duration(
    duration_sec: float,
    *,
    margin: float = 1.2,
    floor: int = 64,
    ceil: int | None = None,
) -> int:
    """Suggest ``max_new_tokens`` from audio length (p99 diar rate + margin).

    Empirics (train+dev sample): diar ~30s p99≈281 tok / ~9.4 tok/s;
    flat ~10s p99≈76. ``margin`` covers denser speech / more speakers.
    """
    if duration_sec <= 0:
        return config.MAX_NEW_TOKENS_CHUNK_30S
    raw = int(config.TOKENS_PER_AUDIO_SEC_P99 * float(duration_sec) * margin)
    # Round up to multiple of 32 (ORT-friendly, less waste than 512 default).
    nice = ((raw + 31) // 32) * 32
    nice = max(floor, nice)
    if ceil is not None:
        nice = min(ceil, nice)
    return nice


def max_new_tokens_for_chunk(chunk_sec: int | float) -> int:
    """Preset for fixed chunking (10s / 30s pipelines)."""
    c = float(chunk_sec)
    if c <= 12:
        return config.MAX_NEW_TOKENS_CHUNK_10S
    if c <= 35:
        return config.MAX_NEW_TOKENS_CHUNK_30S
    return max_new_tokens_for_duration(c)

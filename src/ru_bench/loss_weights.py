"""Per-token CE weights for MOSS compact targets (speaker / timestamp / text)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

_BRACKET_RE = re.compile(r"\[([^\]]*)\]")
_SPEAKER_RE = re.compile(r"^S\d+$", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"^\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class LossWeightConfig:
    speaker: float = 4.0
    timestamp: float = 2.0
    text: float = 1.0
    # Downweight trivial Golos wrap ``[0.00][S01]…[dur]``.
    flat_wrap: float = 0.4
    # Multiply all token weights on EN-domain clips (anti-forgetting).
    en_clip: float = 1.5


def _bracket_spans(target: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for m in _BRACKET_RE.finditer(target):
        content = m.group(1)
        if _SPEAKER_RE.fullmatch(content):
            kind = "speaker"
        elif _TIMESTAMP_RE.fullmatch(content):
            kind = "timestamp"
        else:
            kind = "other"
        spans.append((m.start(), m.end(), kind))
    return spans


def _flat_wrap_char_ranges(spans: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    ts = [(s, e) for s, e, k in spans if k == "timestamp"]
    spk = [(s, e) for s, e, k in spans if k == "speaker"]
    ranges: list[tuple[int, int]] = []
    if ts:
        ranges.append(ts[0])
        if len(ts) > 1:
            ranges.append(ts[-1])
    if spk:
        ranges.append(spk[0])
    return ranges


def _kind_at(char_start: int, char_end: int, spans: list[tuple[int, int, str]]) -> str | None:
    if char_start == char_end:
        return None
    mid = (char_start + char_end) // 2
    for s, e, kind in spans:
        if s <= mid < e:
            return kind
    return None


def _overlaps(a0: int, a1: int, ranges: list[tuple[int, int]]) -> bool:
    for s, e in ranges:
        if not (a1 <= s or a0 >= e):
            return True
    return False


def target_token_weights(
    tokenizer,
    target: str,
    *,
    cfg: LossWeightConfig,
    is_flat: bool,
) -> list[float]:
    """Weights aligned with ``tokenizer(target, add_special_tokens=False)`` tokens."""
    enc = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    spans = _bracket_spans(target)
    wrap = _flat_wrap_char_ranges(spans) if is_flat else []

    weights: list[float] = []
    for start, end in offsets:
        kind = _kind_at(start, end, spans)
        if kind == "speaker":
            w = cfg.flat_wrap if is_flat and _overlaps(start, end, wrap) else cfg.speaker
        elif kind == "timestamp":
            w = cfg.flat_wrap if is_flat and _overlaps(start, end, wrap) else cfg.timestamp
        else:
            w = cfg.text
        weights.append(float(w))
    return weights


def align_label_weights(
    *,
    labels: torch.Tensor,
    prompt_len: int,
    target_weights: list[float],
    eos_weight: float = 1.0,
) -> torch.Tensor:
    """Build ``[1, seq]`` weights; prompt 0, target uses ``target_weights``, then eos."""
    del prompt_len
    weights = torch.zeros_like(labels, dtype=torch.float32)
    active = (labels[0] != -100).nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        return weights

    n_active = int(active.numel())
    n_tgt = len(target_weights)
    if n_active == n_tgt + 1:
        for i, w in enumerate(target_weights):
            weights[0, int(active[i])] = w
        weights[0, int(active[-1])] = float(eos_weight)
    elif n_active == n_tgt:
        for i, w in enumerate(target_weights):
            weights[0, int(active[i])] = w
    else:
        weights[0, active] = 1.0
    return weights


def weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_weights: torch.Tensor,
) -> torch.Tensor:
    """Token-weighted causal LM CE (prompt already masked with -100 / weight 0)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_w = label_weights[:, 1:].contiguous()

    vocab = shift_logits.size(-1)
    flat_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    mask = shift_labels.view(-1) != -100
    w = shift_w.view(-1) * mask.to(dtype=shift_w.dtype)
    denom = w.sum().clamp_min(1e-6)
    return (flat_loss * w).sum() / denom

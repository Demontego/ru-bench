"""Pad a list of single-example training dicts into one microbatch."""

from __future__ import annotations

import torch


def _pad_1d(tensors: list[torch.Tensor], pad_value, max_seq: int) -> torch.Tensor:
    out = []
    for t in tensors:
        pad_len = max_seq - t.shape[0]
        if pad_len:
            out.append(torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)]))
        else:
            out.append(t)
    return torch.stack(out)


def collate_training_examples(examples: list[dict]) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("empty batch")
    if len(examples) == 1:
        return {k: v for k, v in examples[0].items()}

    batch_input_ids = [ex["input_ids"].squeeze(0) for ex in examples]
    batch_attention = [ex["attention_mask"].squeeze(0) for ex in examples]
    batch_labels = [ex["labels"].squeeze(0) for ex in examples]
    batch_lw = [ex["label_weights"].squeeze(0) for ex in examples]

    max_seq = max(ids.shape[0] for ids in batch_input_ids)

    input_features = torch.cat([ex["input_features"] for ex in examples], dim=0)
    audio_feature_lengths = torch.cat(
        [ex["audio_feature_lengths"] for ex in examples], dim=0
    )
    audio_chunk_mapping = torch.cat(
        [
            torch.full_like(ex["audio_chunk_mapping"], fill_value=i)
            for i, ex in enumerate(examples)
        ],
        dim=0,
    )

    return {
        "input_ids": _pad_1d(batch_input_ids, 0, max_seq),
        "attention_mask": _pad_1d(batch_attention, 0, max_seq),
        "labels": _pad_1d(batch_labels, -100, max_seq),
        "label_weights": _pad_1d(batch_lw, 0.0, max_seq),
        "input_features": input_features,
        "audio_feature_lengths": audio_feature_lengths,
        "audio_chunk_mapping": audio_chunk_mapping,
    }

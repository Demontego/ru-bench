"""Patch Whisper audio ONNX so ``mel_len`` can be < 3000 (10s chunks).

Exported graphs bake a fixed positional table ``[1500, H]`` and ``Add`` it to
conv output. Shorter mel → seq≠1500 → ORT broadcast fail. Replace with
``Slice(pos, 0:seq)`` so 10s (mel≈1000 → seq=500) works.

Also: ``trim_mel_features`` cuts pad zeros after the frames needed for
``audio_feature_lengths * merge_size`` (Whisper stride-2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def mel_frames_for_audio_tokens(
    n_audio_tokens: int,
    *,
    merge_size: int = 4,
    whisper_stride: int = 2,
    max_mel: int = 3000,
    margin_frames: int = 0,
) -> int:
    """Minimum mel frames so encoder seq covers ``n_audio_tokens * merge_size``."""
    need_enc = int(n_audio_tokens) * int(merge_size)
    need_mel = need_enc * int(whisper_stride) + int(margin_frames)
    # Even length preferred (stride-2).
    if need_mel % whisper_stride:
        need_mel += whisper_stride - (need_mel % whisper_stride)
    return int(min(max(need_mel, whisper_stride), max_mel))


def trim_mel_features(
    input_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    *,
    merge_size: int = 4,
    max_mel: int = 3000,
    margin_frames: int = 0,
) -> np.ndarray:
    """``input_features`` [B, 80, T] → trimmed [B, 80, T'] (contiguous)."""
    n = int(np.max(audio_feature_lengths.astype(np.int64)))
    t_need = mel_frames_for_audio_tokens(
        n, merge_size=merge_size, max_mel=max_mel, margin_frames=margin_frames
    )
    t_cur = int(input_features.shape[-1])
    if t_cur <= t_need:
        return np.ascontiguousarray(input_features)
    return np.ascontiguousarray(input_features[..., :t_need])


def patch_audio_onnx_dynamic_pos(
    src: Path,
    dst: Path,
    *,
    pos_init_name: str = "onnx::Add_2488",
    add_node_name: str = "/whisper/Add_2",
) -> Path:
    """Write ``dst`` with sliced Whisper positional embeddings."""
    model = onnx.load(str(src), load_external_data=True)
    add = next((n for n in model.graph.node if n.name == add_node_name), None)
    if add is None or add.op_type != "Add":
        raise RuntimeError(f"Add node {add_node_name!r} not found in {src}")
    if pos_init_name not in add.input:
        raise RuntimeError(
            f"{add_node_name} inputs={list(add.input)} missing {pos_init_name!r}"
        )
    x_name = next(i for i in add.input if i != pos_init_name)

    # Constants (opset 17: Unsqueeze axes is an input, not attribute)
    starts_name = "dynpos_starts"
    axes_name = "dynpos_axes"
    unsqueeze_axes = "dynpos_unsqueeze_axes"
    gather_idx = "dynpos_gather_idx"
    for name, vals in (
        (starts_name, [0]),
        (axes_name, [0]),
        (unsqueeze_axes, [0]),
        (gather_idx, 1),
    ):
        if not any(t.name == name for t in model.graph.initializer):
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(vals, dtype=np.int64), name=name)
            )

    shape_out = "dynpos_shape"
    seq_out = "dynpos_seq"
    ends_out = "dynpos_ends"
    pos_sliced = "dynpos_pos_sliced"

    new_nodes = [
        helper.make_node("Shape", [x_name], [shape_out], name="dynpos_Shape"),
        helper.make_node(
            "Gather",
            [shape_out, gather_idx],
            [seq_out],
            name="dynpos_Gather_seq",
            axis=0,
        ),
        helper.make_node(
            "Unsqueeze",
            [seq_out, unsqueeze_axes],
            [ends_out],
            name="dynpos_Unsqueeze_ends",
        ),
        helper.make_node(
            "Slice",
            [pos_init_name, starts_name, ends_out, axes_name],
            [pos_sliced],
            name="dynpos_Slice_pos",
        ),
    ]
    # Rewire Add to use sliced pos
    add.input[list(add.input).index(pos_init_name)] = pos_sliced

    # Insert nodes before Add
    idx = next(i for i, n in enumerate(model.graph.node) if n.name == add_node_name)
    for j, node in enumerate(new_nodes):
        model.graph.node.insert(idx + j, node)

    dst.parent.mkdir(parents=True, exist_ok=True)
    # External data: keep next to src if large
    if src.resolve().parent == dst.resolve().parent:
        onnx.save_model(
            model,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=dst.name + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(model, str(dst))
    return dst

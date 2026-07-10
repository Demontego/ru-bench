"""Split ONNX export: audio once + embed + LM prefill/decode with KV-cache.

Graphs
------
- ``audio.onnx``       : mel → audio_embeds ``[N, H]``
- ``embed.onnx``       : input_ids → token embeds ``[B, S, H]``
- ``lm_prefill.onnx``  : full prompt embeds → logits + present_kv (no past)
- ``lm_decode.onnx``   : last-token embed + past_kv → logits + present_kv
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, DynamicCache

from ru_bench.model_runner import MODEL_ID
from ru_bench.onnx_export import (
    export_extensions_tokenizer,
    load_merged_model,
    optimize_with_ort_transformers,
)

N_LAYERS = 28
N_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = 1024


def pack_cache(cache: DynamicCache) -> tuple[torch.Tensor, ...]:
    out: list[torch.Tensor] = []
    for layer in cache.layers:
        out.append(layer.keys.contiguous())
        out.append(layer.values.contiguous())
    return tuple(out)


def unpack_cache(flat: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> DynamicCache:
    cache = DynamicCache()
    for i in range(N_LAYERS):
        cache.update(flat[2 * i], flat[2 * i + 1], i)
    return cache


def empty_past(
    batch: int = 1,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    device = device or torch.device("cpu")
    return tuple(
        torch.zeros(batch, N_KV_HEADS, 0, HEAD_DIM, device=device, dtype=dtype)
        for _ in range(N_LAYERS * 2)
    )


def past_names() -> list[str]:
    names: list[str] = []
    for i in range(N_LAYERS):
        names.append(f"past_key_{i}")
        names.append(f"past_value_{i}")
    return names


def present_names() -> list[str]:
    names: list[str] = []
    for i in range(N_LAYERS):
        names.append(f"present_key_{i}")
        names.append(f"present_value_{i}")
    return names


class MossOnnxAudio(nn.Module):
    """Whisper + time_merge + VQAdaptor → audio token embeds."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.whisper = core.whisper_encoder
        self.time_merge = core.time_merge
        self.vq = core.vq_adaptor
        self.merge_size = int(core.config.audio_merge_size)

    def forward(
        self,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        enc = self.whisper(input_features, return_dict=True).last_hidden_state
        n = audio_feature_lengths[0]
        feat = enc[:, : n * self.merge_size, :].to(dtype=enc.dtype)
        merged = self.time_merge(feat)
        return self.vq(merged).squeeze(0)  # [N, H]


class MossOnnxEmbed(nn.Module):
    def __init__(self, embed: nn.Module) -> None:
        super().__init__()
        self.embed = embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class MossOnnxLMPrefill(nn.Module):
    """Full-prompt LM step; no past inputs (exports clean present KV)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state[:, -1:, :])
        return (logits,) + pack_cache(out.past_key_values)


class MossOnnxLMDecode(nn.Module):
    """Single-token LM step: ``input_ids`` → embed → LM + flat KV IO.

    Trace with ``past_len>0``. Embedding is inside the graph so the host decode
    loop does one ORT run per token (no separate embed session).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.embed = model.model.get_input_embeddings()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        inputs_embeds = self.embed(input_ids)
        cache = unpack_cache(past)
        out = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state[:, -1:, :])
        return (logits,) + pack_cache(out.past_key_values)


# Back-compat alias
MossOnnxLM = MossOnnxLMDecode


def scatter_audio_embeds(
    token_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_embeds: torch.Tensor,
    audio_token_id: int,
) -> torch.Tensor:
    """Host-side (numpy/torch): replace audio placeholder positions."""
    mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(token_embeds)
    return token_embeds.masked_scatter(mask, audio_embeds.to(dtype=token_embeds.dtype))


def _save_external(onnx_path: Path, work_raw: Path) -> Path:
    from onnx import load as onnx_load
    from onnx import save_model

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    data_name = f"{onnx_path.name}.data"
    onnx_path.unlink(missing_ok=True)
    (onnx_path.parent / data_name).unlink(missing_ok=True)
    model = onnx_load(str(work_raw), load_external_data=True)
    save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return onnx_path


def _export_module(
    module: nn.Module,
    args: tuple[Any, ...],
    *,
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
    work: Path,
) -> Path:
    raw = work / f"{path.stem}_raw.onnx"
    torch.onnx.export(
        module,
        args,
        str(raw),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    return _save_external(path, raw)


def export_onnx_kv(
    *,
    checkpoint_dir: str | Path | None,
    output_dir: str | Path,
    sample_audio: str | Path,
    opset: int = 17,
    optimize_lm: bool = False,
    float16: bool = False,
    use_gpu: bool = False,
    export_tokenizer: bool = True,
) -> dict[str, Path]:
    """Export audio / embed / lm KV graphs into ``output_dir``."""
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        prepare_inputs,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work = output_dir / "_onnx_kv_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    device = torch.device("cpu")
    model = load_merged_model(checkpoint_dir, device)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = build_transcription_messages(str(sample_audio))
    inputs = prepare_inputs(processor, messages, device=device)

    audio_token_id = int(model.config.audio_token_id)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_features = inputs["input_features"]
    audio_feature_lengths = inputs["audio_feature_lengths"]

    # --- audio ---
    audio_mod = MossOnnxAudio(model.model).eval()
    with torch.no_grad():
        audio_embeds = audio_mod(input_features, audio_feature_lengths)
    audio_path = _export_module(
        audio_mod,
        (input_features, audio_feature_lengths),
        path=output_dir / "audio.onnx",
        input_names=["input_features", "audio_feature_lengths"],
        output_names=["audio_embeds"],
        dynamic_axes={
            "input_features": {0: "chunks", 2: "mel_len"},
            "audio_feature_lengths": {0: "chunks"},
            "audio_embeds": {0: "n_audio"},
        },
        opset=opset,
        work=work,
    )

    # --- embed ---
    embed_mod = MossOnnxEmbed(model.model.get_input_embeddings()).eval()
    embed_path = _export_module(
        embed_mod,
        (input_ids,),
        path=output_dir / "embed.onnx",
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "inputs_embeds": {0: "batch", 1: "seq"},
        },
        opset=opset,
        work=work,
    )

    # --- lm_prefill (no past IO) ---
    prefill_mod = MossOnnxLMPrefill(model).eval()
    with torch.no_grad():
        tok_embeds = embed_mod(input_ids)
        pref_embeds = scatter_audio_embeds(
            tok_embeds, input_ids, audio_embeds, audio_token_id
        )
        pref_out = prefill_mod(pref_embeds, attention_mask)
        first_tok = int(pref_out[0][0, -1].argmax().item())
        past_after = pref_out[1:]

    prefill_path = _export_module(
        prefill_mod,
        (pref_embeds, attention_mask),
        path=output_dir / "lm_prefill.onnx",
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits"] + present_names(),
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "total"},
            "logits": {0: "batch", 1: "out_seq"},
            **{
                f"present_key_{i}": {0: "batch", 2: "present"}
                for i in range(N_LAYERS)
            },
            **{
                f"present_value_{i}": {0: "batch", 2: "present"}
                for i in range(N_LAYERS)
            },
        },
        opset=opset,
        work=work,
    )

    # --- lm_decode (past_len>0 trace; embed fused — input_ids in) ---
    decode_mod = MossOnnxLMDecode(model).eval()
    with torch.no_grad():
        step_ids = torch.tensor([[first_tok]], dtype=input_ids.dtype)
        step_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)], dim=1
        )
    decode_args = (step_ids, step_mask) + past_after
    in_names = ["input_ids", "attention_mask"] + past_names()
    out_names = ["logits"] + present_names()
    dyn: dict[str, dict[int, str]] = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "total"},
        "logits": {0: "batch", 1: "out_seq"},
    }
    for i in range(N_LAYERS):
        dyn[f"past_key_{i}"] = {0: "batch", 2: "past"}
        dyn[f"past_value_{i}"] = {0: "batch", 2: "past"}
        dyn[f"present_key_{i}"] = {0: "batch", 2: "present"}
        dyn[f"present_value_{i}"] = {0: "batch", 2: "present"}

    decode_path = _export_module(
        decode_mod,
        decode_args,
        path=output_dir / "lm_decode.onnx",
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dyn,
        opset=opset,
        work=work,
    )

    shutil.rmtree(work, ignore_errors=True)

    processor_dir = output_dir / "processor"
    if processor_dir.exists():
        shutil.rmtree(processor_dir)
    processor.save_pretrained(processor_dir)
    AutoTokenizer.from_pretrained(processor_dir, trust_remote_code=True).save_pretrained(
        processor_dir
    )

    meta = {
        "audio_token_id": audio_token_id,
        "n_layers": N_LAYERS,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "graphs": ["audio.onnx", "embed.onnx", "lm_prefill.onnx", "lm_decode.onnx"],
        "lm_decode_inputs": "input_ids",  # embed fused inside decode
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
    }
    meta_path = output_dir / "kv_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    artifacts: dict[str, Path] = {
        "audio": audio_path,
        "embed": embed_path,
        "lm_prefill": prefill_path,
        "lm_decode": decode_path,
        "processor": processor_dir,
        "meta": meta_path,
    }

    if optimize_lm:
        for src_key, stem in (("lm_prefill", "lm_prefill"), ("lm_decode", "lm_decode")):
            src = artifacts[src_key]
            opt_path = src.with_name(f"{stem}.opt.onnx")
            optimize_with_ort_transformers(
                src, opt_path, float16=float16, use_gpu=use_gpu, model_type="gpt2"
            )
            artifacts[f"{src_key}_opt"] = opt_path

    if export_tokenizer:
        tok_meta = export_extensions_tokenizer(
            processor_dir, output_dir / "extensions_tokenizer.json"
        )
        artifacts["tokenizer_ext"] = tok_meta

    return artifacts

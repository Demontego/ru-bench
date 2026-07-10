"""ONNX one-step export for merged MOSS-Transcribe-Diarize (+ optional LoRA).

Post-export:
  - ``onnxruntime.transformers.optimizer`` (Attention/GELU/LayerNorm fusions)
  - ``onnxruntime-extensions`` HfJsonTokenizer graph for vocab encode
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from ru_bench.model_runner import MODEL_ID


def _patch_audio_path_for_onnx(core: nn.Module) -> None:
    """Replace Python-loop audio path with batch=1 tensor ops (trace/ONNX friendly)."""
    merge_size = int(core.config.audio_merge_size)

    def get_audio_features(
        input_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
        audio_chunk_mapping: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        del audio_chunk_mapping  # single-chunk export assumes mapping=[0]
        whisper_features = core.whisper_encoder(input_features, return_dict=True).last_hidden_state
        n_tok = audio_feature_lengths[0]
        feat = whisper_features[:, : n_tok * merge_size, :].to(dtype=core.dtype)
        merged = core.time_merge(feat)
        return [core.vq_adaptor(merged)]

    def get_placeholder_mask(
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        del audio_features  # length validated by processor at preprocess time
        assert input_ids is not None
        special_audio_mask = input_ids.to(device=inputs_embeds.device) == core.config.audio_token_id
        return special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds)

    core.get_audio_features = get_audio_features  # type: ignore[method-assign]
    core.get_placeholder_mask = get_placeholder_mask  # type: ignore[method-assign]


class MossOnnxStep(nn.Module):
    """Single decode step → last-token logits (logits_to_keep=1)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        _patch_audio_path_for_onnx(model.model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        audio_feature_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch = input_features.shape[0]
        audio_chunk_mapping = torch.zeros(batch, dtype=torch.long, device=input_ids.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_chunk_mapping=audio_chunk_mapping,
            use_cache=False,
            logits_to_keep=1,
        )
        return out.logits


def load_merged_model(checkpoint_dir: str | Path | None, device: torch.device) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float32
    )
    if checkpoint_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
        model = model.merge_and_unload()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model.to(device).eval()


def _qwen_optimizer_dims() -> tuple[int, int]:
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text = cfg.text_config
    return int(text.num_attention_heads), int(text.hidden_size)


def optimize_with_ort_transformers(
    input_onnx: Path,
    output_onnx: Path,
    *,
    float16: bool = False,
    use_gpu: bool = False,
    opt_level: int | None = 0,
    model_type: str = "gpt2",
) -> Path:
    """Offline graph fusions via ``onnxruntime.transformers.optimizer``.

    Docs: https://onnxruntime.ai/docs/performance/transformers-optimization.html

    ``model_type='gpt2'`` is closest built-in match for Qwen decoder Attention.
    Default ``opt_level=0``: fusion-only. ``opt_level>=1`` breaks this hybrid
    Whisper+Qwen export (duplicate Constant initializers / invalid ``If`` nodes).
    """
    from onnxruntime.transformers import optimizer

    num_heads, hidden_size = _qwen_optimizer_dims()
    optimized = optimizer.optimize_model(
        str(input_onnx),
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=opt_level,
        use_gpu=use_gpu,
        only_onnxruntime=False,
        verbose=False,
    )
    if float16:
        optimized.convert_float_to_float16(keep_io_types=True)

    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    # Drop previous packed artifacts (stem.opt.onnx + stem.opt.onnx.data).
    for p in output_onnx.parent.glob(output_onnx.name + "*"):
        if p.is_file():
            p.unlink(missing_ok=True)
    optimized.save_model_to_file(
        str(output_onnx),
        use_external_data_format=True,
        all_tensors_to_one_file=True,
    )
    return output_onnx


def quantize_dynamic_int8(
    input_onnx: Path,
    output_onnx: Path,
    *,
    per_channel: bool = True,
    reduce_range: bool = False,
    weight_type: str = "QInt8",
    op_types_to_quantize: tuple[str, ...] | None = None,
    nodes_to_exclude: list[str] | None = None,
) -> Path:
    """CPU-oriented dynamic INT8 (weights INT8, activations quantized at runtime).

    Prefer this for ``CPUExecutionProvider``. Uses ORT ``quantize_dynamic`` —
    good fit for MatMul-heavy decoder graphs without a calibration set.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_onnx = Path(input_onnx)
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    for p in output_onnx.parent.glob(output_onnx.name + "*"):
        if p.is_file():
            p.unlink(missing_ok=True)

    wt = getattr(QuantType, weight_type)
    quantize_dynamic(
        model_input=str(input_onnx),
        model_output=str(output_onnx),
        op_types_to_quantize=list(op_types_to_quantize)
        if op_types_to_quantize is not None
        else ["MatMul", "Gemm"],
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=wt,
        nodes_to_exclude=nodes_to_exclude,
        use_external_data_format=True,
        extra_options={"DefaultTensorType": 1},  # FLOAT
    )
    return output_onnx


def quantize_matmul_nbits(
    input_onnx: Path,
    output_onnx: Path,
    *,
    bits: int = 4,
    block_size: int = 128,
    is_symmetric: bool = False,
    accuracy_level: int | None = 4,
) -> Path:
    """Weight-only MatMul quantization (Q4 / INT8) via ORT ``MatMulNBitsQuantizer``.

    Keeps activations FP; packs constant MatMul weights into ``MatMulNBits``.
    Prefer running on ``*.opt.onnx`` (already fused). ``accuracy_level=4`` uses
    higher-precision accumulation for 4-bit kernels when available.
    """
    if bits not in {2, 4, 8}:
        raise ValueError(f"bits must be 2/4/8, got {bits}")

    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )
    from onnxruntime.quantization.quant_utils import QuantFormat

    input_onnx = Path(input_onnx)
    output_onnx = Path(output_onnx)
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    for p in output_onnx.parent.glob(output_onnx.name + "*"):
        if p.is_file():
            p.unlink(missing_ok=True)

    model = onnx.load(str(input_onnx), load_external_data=True)
    cfg = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=accuracy_level,
        quant_format=QuantFormat.QOperator,
        bits=bits,
    )
    quant = MatMulNBitsQuantizer(
        model=model,
        bits=bits,
        accuracy_level=accuracy_level,
        algo_config=cfg,
    )
    quant.process()
    quant.model.save_model_to_file(str(output_onnx), use_external_data_format=True)
    return output_onnx


def export_extensions_tokenizer(processor_dir: Path, output_meta: Path) -> Path:
    """Validate + document Extensions tokenizer for this processor dir.

    Qwen2 ``tokenizer.json`` is not loadable by the ``HfJsonTokenizer`` custom op
    (``Failed to parse config json``). Use ``onnxruntime_extensions.pp_api.Tokenizer``
    instead — same Extensions package, supports ``tokenize`` / ``apply_chat_template``.

    Docs: https://onnxruntime.ai/docs/extensions/
    """
    from onnxruntime_extensions.pp_api import Tokenizer

    processor_dir = Path(processor_dir).resolve()
    if not (processor_dir / "tokenizer.json").exists():
        AutoTokenizer.from_pretrained(processor_dir, trust_remote_code=True).save_pretrained(
            processor_dir
        )
    tok = Tokenizer(str(processor_dir))
    probe = tok.tokenize(["ping"])
    if not probe or not probe[0]:
        raise RuntimeError("pp_api.Tokenizer.tokenize returned empty ids")

    output_meta = Path(output_meta)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "api": "onnxruntime_extensions.pp_api.Tokenizer",
        "processor_dir": str(processor_dir),
        "probe_ids": probe[0][:8],
        "note": (
            "HfJsonTokenizer ONNX op cannot parse Qwen2 tokenizer.json; "
            "use pp_api.Tokenizer(processor_dir) at runtime with "
            "SessionOptions.register_custom_ops_library(get_library_path())."
        ),
        "docs": "https://onnxruntime.ai/docs/extensions/",
    }
    output_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_meta


# Back-compat alias
def export_tokenizer_onnx(processor_dir: Path, output_onnx: Path) -> Path:
    """Deprecated name — writes Extensions tokenizer metadata JSON next to exports."""
    meta_path = Path(output_onnx).with_suffix(".json")
    if meta_path.name.endswith(".onnx.json"):
        meta_path = Path(output_onnx).parent / "extensions_tokenizer.json"
    # Prefer stable name.
    meta_path = Path(output_onnx).parent / "extensions_tokenizer.json"
    return export_extensions_tokenizer(processor_dir, meta_path)


def export_onnx_step(
    *,
    checkpoint_dir: str | Path | None,
    output_path: str | Path,
    sample_audio: str | Path,
    opset: int = 17,
    optimize: bool = True,
    float16: bool = False,
    use_gpu: bool = False,
    export_tokenizer: bool = True,
) -> dict[str, Path]:
    """Export one-step logits graph + optional ORT optimize + tokenizer ONNX."""
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        prepare_inputs,
    )
    from onnx import load as onnx_load
    from onnx import save_model

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = output_path.parent / "_onnx_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    device = torch.device("cpu")
    model = load_merged_model(checkpoint_dir, device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    messages = build_transcription_messages(str(sample_audio))
    inputs = prepare_inputs(processor, messages, device=device)

    wrapper = MossOnnxStep(model)
    args: tuple[Any, ...] = (
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["input_features"],
        inputs["audio_feature_lengths"],
    )
    raw_onnx = work / "raw.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(raw_onnx),
        input_names=[
            "input_ids",
            "attention_mask",
            "input_features",
            "audio_feature_lengths",
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "input_features": {0: "batch"},
            "audio_feature_lengths": {0: "batch"},
            "logits": {0: "batch", 1: "out_seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx_load(str(raw_onnx), load_external_data=True)
    data_name = f"{output_path.name}.data"
    output_path.unlink(missing_ok=True)
    (output_path.parent / data_name).unlink(missing_ok=True)

    save_model(
        onnx_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    shutil.rmtree(work, ignore_errors=True)

    processor_dir = output_path.parent / "processor"
    if processor_dir.exists():
        shutil.rmtree(processor_dir)
    processor.save_pretrained(processor_dir)
    # Ensure tokenizer.json present for Extensions HfJsonTokenizer.
    AutoTokenizer.from_pretrained(processor_dir, trust_remote_code=True).save_pretrained(
        processor_dir
    )

    artifacts: dict[str, Path] = {"onnx": output_path, "processor": processor_dir}

    if optimize:
        opt_path = output_path.with_name(output_path.stem + ".opt.onnx")
        optimize_with_ort_transformers(
            output_path,
            opt_path,
            float16=float16,
            use_gpu=use_gpu,
        )
        artifacts["onnx_opt"] = opt_path

    if export_tokenizer:
        tok_meta = export_extensions_tokenizer(
            processor_dir, output_path.parent / "extensions_tokenizer.json"
        )
        artifacts["tokenizer_ext"] = tok_meta

    return artifacts

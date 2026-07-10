import torch
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"


def require_cuda(device: torch.device | None = None) -> torch.device:
    """Force CUDA for all inference/train runs. Fail loud if GPU torch missing."""
    if device is not None:
        if device.type != "cuda":
            raise RuntimeError(f"CUDA required, got device={device}")
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required but torch.cuda.is_available() is False. "
            f"torch={torch.__version__} — install CUDA wheel (see pyproject pytorch-cu130 index)."
        )
    return resolve_device("cuda")


def load_model(adapter_path: str | None = None, device: torch.device | None = None):
    device = require_cuda(device)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model = model.to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor, device, dtype


def run_one(model, processor, device, dtype, audio_path: str, max_new_tokens: int) -> dict:
    messages = build_transcription_messages(audio_path)
    result = generate_transcription(
        model, processor, messages, device=device, dtype=dtype, max_new_tokens=max_new_tokens
    )
    segments = parse_transcript(result["text"])
    speakers = {s.speaker for s in segments}
    if len(speakers) > 1:
        print(f"warning: {audio_path} is single-speaker but model emitted {len(speakers)} speaker labels")
    merged_text = " ".join(s.text for s in sorted(segments, key=lambda s: s.start))
    return {
        "audio_path": audio_path,
        "raw_text": result["text"],
        "merged_text": merged_text,
        "segments": [
            {"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in segments
        ],
    }

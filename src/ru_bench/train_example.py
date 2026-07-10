from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, process_audio_info

from ru_bench import config
from ru_bench.data import ClipRef
from ru_bench.loss_weights import LossWeightConfig, align_label_weights, target_token_weights
from ru_bench.moss_format import target_text


def build_training_example(
    clip: ClipRef,
    processor,
    prompt: str = DEFAULT_PROMPT,
    *,
    loss_weights: LossWeightConfig | None = None,
) -> dict:
    """Tokenize one clip into model inputs + labels + CE token weights."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": clip.audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    tgt = target_text(clip)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + tgt + processor.tokenizer.eos_token

    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)

    prompt_inputs = processor(text=prompt_text, audio=audios, return_tensors="pt")
    full_inputs = processor(text=full_text, audio=audios, return_tensors="pt")

    prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())
    labels = full_inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100
    full_inputs["labels"] = labels

    cfg = loss_weights or LossWeightConfig()
    is_flat = clip.target is None
    tw = target_token_weights(processor.tokenizer, tgt, cfg=cfg, is_flat=is_flat)
    if clip.domain in config.EN_DOMAINS and cfg.en_clip != 1.0:
        tw = [w * cfg.en_clip for w in tw]
    full_inputs["label_weights"] = align_label_weights(
        labels=labels,
        prompt_len=prompt_len,
        target_weights=tw,
    )
    return full_inputs

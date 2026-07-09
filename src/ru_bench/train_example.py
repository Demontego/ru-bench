from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT, process_audio_info

from ru_bench.data import ClipRef


def target_text(clip: ClipRef) -> str:
    return f"[0.00][S01]{clip.text}[{clip.duration:.2f}]"


def build_training_example(clip: ClipRef, processor, prompt: str = DEFAULT_PROMPT) -> dict:
    """Tokenize one Golos clip into model inputs + labels for SFT.

    Two-pass tokenization: encode the prompt alone to find where it ends,
    then encode prompt+target together and mask everything before that
    boundary with -100. The target is appended directly after the prompt
    (no <think> block) to match what the base model actually produces at
    inference (verified empirically: none of our saved MOSS outputs contain
    "<think>", even though the tokenizer's chat template inserts an empty
    one when rendering stored assistant turns).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": clip.audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + target_text(clip) + processor.tokenizer.eos_token

    audios = process_audio_info(messages, sampling_rate=processor.feature_extractor.sampling_rate)

    prompt_inputs = processor(text=prompt_text, audio=audios, return_tensors="pt")
    full_inputs = processor(text=full_text, audio=audios, return_tensors="pt")

    prompt_len = int(prompt_inputs["attention_mask"][0].sum().item())
    labels = full_inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100

    full_inputs["labels"] = labels
    return full_inputs

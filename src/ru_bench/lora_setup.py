from peft import LoraConfig, get_peft_model

from ru_bench import config

# Leaf names shared/unique across components:
#   Whisper self-attn: q_proj, k_proj, v_proj, out_proj   Whisper FFN: fc1, fc2
#   Qwen3 self-attn:    q_proj, k_proj, v_proj, o_proj     Qwen3 FFN:  gate_proj, up_proj, down_proj (excluded -> "light")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "fc1", "fc2"]

# rank/alpha overrides are matched by regex against each module's full dotted path,
# so q_proj/k_proj/v_proj (name shared by both components) still get different
# strength LoRA depending on which submodule tree they live in.
RANK_PATTERN = {
    r".*whisper_encoder.*": config.LORA_R_WHISPER,
    r".*language_model.*": config.LORA_R_QWEN,
}
ALPHA_PATTERN = {
    r".*whisper_encoder.*": config.LORA_ALPHA_WHISPER,
    r".*language_model.*": config.LORA_ALPHA_QWEN,
}


def build_lora_model(base_model):
    lora_config = LoraConfig(
        r=config.LORA_R_WHISPER,
        lora_alpha=config.LORA_ALPHA_WHISPER,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        rank_pattern=RANK_PATTERN,
        alpha_pattern=ALPHA_PATTERN,
        modules_to_save=["vq_adaptor"],
        bias="none",
    )
    return get_peft_model(base_model, lora_config)


def trainable_parameter_summary(model) -> dict:
    trainable, total = 0, 0
    by_group = {"whisper_lora": 0, "qwen_lora": 0, "vq_adaptor": 0, "other": 0}
    for name, param in model.named_parameters():
        total += param.numel()
        if not param.requires_grad:
            continue
        trainable += param.numel()
        if "vq_adaptor" in name:
            by_group["vq_adaptor"] += param.numel()
        elif "whisper_encoder" in name:
            by_group["whisper_lora"] += param.numel()
        elif "language_model" in name:
            by_group["qwen_lora"] += param.numel()
        else:
            by_group["other"] += param.numel()
    return {"trainable": trainable, "total": total, "by_group": by_group}

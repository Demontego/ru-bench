import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ru_bench import config
from ru_bench.data import load_manifest
from ru_bench.lora_setup import build_lora_model, trainable_parameter_summary
from ru_bench.loss_weights import LossWeightConfig
from ru_bench.model_runner import MODEL_ID
from ru_bench.train_loop import resolve_train_device, train


def load_train_model(resume_from: str | None):
    """Fresh LoRA on base, or continue from an existing PEFT adapter dir."""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
    if resume_from:
        from peft import PeftModel

        path = Path(resume_from)
        if not path.exists():
            raise SystemExit(f"--resume-from not found: {path}")
        print(f"Resuming LoRA adapter from {path}")
        model = PeftModel.from_pretrained(base_model, str(path), is_trainable=True)
    else:
        model = build_lora_model(base_model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default=str(config.TRAIN_MANIFEST_PATH))
    parser.add_argument(
        "--dev-manifest",
        default=str(config.DEV_MANIFEST_PATH),
        help="path to dev manifest; pass 'none' to skip eval",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap on train clips")
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000, help="microbatch forwards")
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--batch-accum", type=int, default=config.TRAIN_BATCH_ACCUM)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=600)
    parser.add_argument("--eval-max-clips", type=int, default=32)
    parser.add_argument("--metrics-every", type=int, default=600)
    parser.add_argument("--metrics-max-clips", type=int, default=8)
    parser.add_argument("--metrics-max-new-tokens", type=int, default=512)
    parser.add_argument("--spk-weight", type=float, default=config.LOSS_W_SPEAKER)
    parser.add_argument("--ts-weight", type=float, default=config.LOSS_W_TIMESTAMP)
    parser.add_argument("--text-weight", type=float, default=config.LOSS_W_TEXT)
    parser.add_argument("--flat-wrap-weight", type=float, default=config.LOSS_W_FLAT_WRAP)
    parser.add_argument("--en-clip-weight", type=float, default=config.LOSS_W_EN_CLIP)
    parser.add_argument(
        "--en-sample-ratio",
        type=float,
        default=config.TRAIN_EN_SAMPLE_RATIO,
        help="fraction of microbatch slots drawn from EN (anti-forgetting upsample)",
    )
    parser.add_argument(
        "--en-diar-ratio",
        type=float,
        default=config.TRAIN_EN_DIAR_RATIO,
        help="within EN draws: fraction from multi-spk diar vs flat ASR",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    train_clips = load_manifest(args.train_manifest)
    skip_dev = not args.dev_manifest or args.dev_manifest.lower() in {"none", "null", "-"}
    if args.dev_limit is None and not skip_dev:
        args.dev_limit = max(args.eval_max_clips, args.metrics_max_clips)
    dev_clips = [] if skip_dev else load_manifest(args.dev_manifest)
    if args.limit:
        train_clips = train_clips[: args.limit]
    if args.dev_limit:
        diar = [c for c in dev_clips if c.target and c.target.count("[S") >= 2]
        other = [c for c in dev_clips if c not in diar]
        # Prefer EN diar in metrics pool so retention is visible.
        en_diar = [c for c in diar if c.domain in config.EN_DOMAINS]
        ru_diar = [c for c in diar if c.domain not in config.EN_DOMAINS]
        half = max(1, args.dev_limit // 2)
        picked = (en_diar[:half] + ru_diar + other + en_diar[half:])[: args.dev_limit]
        dev_clips = picked
    print(f"train clips: {len(train_clips)}, dev clips: {len(dev_clips)}", flush=True)

    device = resolve_train_device(args.device)
    if device.type != "cuda":
        raise SystemExit(f"CUDA required for finetune, got device={device}")
    dtype = torch.bfloat16
    eff = args.batch_size * args.batch_accum
    loss_weights = LossWeightConfig(
        speaker=args.spk_weight,
        timestamp=args.ts_weight,
        text=args.text_weight,
        flat_wrap=args.flat_wrap_weight,
        en_clip=args.en_clip_weight,
    )
    print(
        f"device={device}, dtype={dtype}, "
        f"batch={args.batch_size} accum={args.batch_accum} eff={eff} lr={args.lr}",
        flush=True,
    )
    print(
        f"ce_weights spk={loss_weights.speaker} ts={loss_weights.timestamp} "
        f"text={loss_weights.text} flat={loss_weights.flat_wrap} "
        f"en_clip={loss_weights.en_clip} en_sample_ratio={args.en_sample_ratio} "
        f"en_diar_ratio={args.en_diar_ratio}",
        flush=True,
    )
    if args.resume_from:
        print(f"resume_from={args.resume_from}", flush=True)

    model = load_train_model(args.resume_from)
    print(trainable_parameter_summary(model), flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    history = train(
        model,
        processor,
        train_clips,
        dev_clips,
        device=device,
        dtype=dtype,
        steps=args.steps,
        batch_size=args.batch_size,
        batch_accum=args.batch_accum,
        lr=args.lr,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_max_clips=args.eval_max_clips,
        metrics_every=args.metrics_every,
        metrics_max_clips=args.metrics_max_clips,
        metrics_max_new_tokens=args.metrics_max_new_tokens,
        loss_weights=loss_weights,
        en_sample_ratio=args.en_sample_ratio,
        en_diar_ratio=args.en_diar_ratio,
    )

    out_dir = Path(args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"Saved LoRA + adapter checkpoint -> {out_dir}", flush=True)
    if history:
        print(f"Last log: {history[-1]}", flush=True)


if __name__ == "__main__":
    main()

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ru_bench import config
from ru_bench.data import load_manifest
from ru_bench.lora_setup import build_lora_model, trainable_parameter_summary
from ru_bench.model_runner import MODEL_ID
from ru_bench.train_loop import resolve_train_device, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default=str(config.TRAIN_MANIFEST_PATH))
    parser.add_argument("--dev-manifest", default=str(config.DEV_MANIFEST_PATH))
    parser.add_argument("--limit", type=int, default=None, help="cap on train clips, e.g. for a smoke test")
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    parser.add_argument("--checkpoint-dir", default=str(config.CHECKPOINT_DIR))
    args = parser.parse_args()

    train_clips = load_manifest(args.train_manifest)
    dev_clips = load_manifest(args.dev_manifest) if args.dev_manifest else []
    if args.limit:
        train_clips = train_clips[: args.limit]
    if args.dev_limit:
        dev_clips = dev_clips[: args.dev_limit]
    print(f"train clips: {len(train_clips)}, dev clips: {len(dev_clips)}")

    device = resolve_train_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype="auto")
    model = build_lora_model(base_model)
    print(trainable_parameter_summary(model))

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    def on_log(entry):
        print(entry)

    history = train(
        model,
        processor,
        train_clips,
        dev_clips,
        device=device,
        dtype=dtype,
        steps=args.steps,
        batch_accum=args.batch_accum,
        lr=args.lr,
        log_every=args.log_every,
        eval_every=args.eval_every,
        on_log=on_log,
    )

    model.save_pretrained(args.checkpoint_dir)
    print(f"Saved LoRA + adapter checkpoint -> {args.checkpoint_dir}")
    print("Loss history:", history)


if __name__ == "__main__":
    main()

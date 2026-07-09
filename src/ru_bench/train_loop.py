import itertools

import torch

from ru_bench.data import ClipRef
from ru_bench.train_example import build_training_example


def resolve_train_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(example: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for key, value in example.items():
        if key == "input_features":
            out[key] = value.to(device=device, dtype=dtype)
        else:
            out[key] = value.to(device=device)
    return out


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    import contextlib

    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_dev_loss(model, processor, dev_clips: list[ClipRef], device, dtype) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for clip in dev_clips:
        example = move_to_device(build_training_example(clip, processor), device, dtype)
        with _autocast_context(device, dtype):
            out = model(**example)
        total_loss += float(out.loss.item())
        n += 1
    model.train()
    return total_loss / max(n, 1)


def train(
    model,
    processor,
    train_clips: list[ClipRef],
    dev_clips: list[ClipRef],
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    batch_accum: int = 8,
    lr: float = 1e-4,
    log_every: int = 10,
    eval_every: int = 100,
    on_log=None,
) -> list[dict]:
    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    history = []
    running_loss = 0.0
    optimizer.zero_grad()

    cycler = itertools.cycle(train_clips)
    for step in range(1, steps + 1):
        clip = next(cycler)
        example = move_to_device(build_training_example(clip, processor), device, dtype)

        with _autocast_context(device, dtype):
            out = model(**example)
            loss = out.loss / batch_accum

        loss.backward()
        running_loss += float(out.loss.item())

        if step % batch_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            entry = {"step": step, "train_loss": avg_loss}
            running_loss = 0.0
            if step % eval_every == 0 and dev_clips:
                entry["dev_loss"] = evaluate_dev_loss(model, processor, dev_clips, device, dtype)
            history.append(entry)
            if on_log:
                on_log(entry)

    return history

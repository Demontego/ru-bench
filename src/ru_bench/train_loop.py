import itertools
import random
import time

import torch
from tqdm.auto import tqdm

from ru_bench.collate import collate_training_examples
from ru_bench.data import ClipRef
from ru_bench.loss_weights import LossWeightConfig, weighted_lm_loss
from ru_bench.metrics_moss import aggregate_clip_scores, score_clip_hypothesis
from ru_bench.train_data import clip_language
from ru_bench.train_example import build_training_example


def resolve_train_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_META_KEYS = frozenset({"labels", "label_weights"})


def make_en_retention_picker(
    clips: list[ClipRef],
    *,
    en_ratio: float = 0.25,
    seed: int = 42,
):
    """Upsample EN clips so ~``en_ratio`` of draws are English (anti-forgetting)."""
    ru = [c for c in clips if clip_language(c) == "ru"]
    en = [c for c in clips if clip_language(c) == "en"]
    if not en:
        cycle = itertools.cycle(clips)
        return lambda: next(cycle)
    if not ru:
        cycle = itertools.cycle(en)
        return lambda: next(cycle)

    ru_c = itertools.cycle(ru)
    en_c = itertools.cycle(en)
    rng = random.Random(seed)
    ratio = min(max(en_ratio, 0.0), 0.9)

    def pick() -> ClipRef:
        return next(en_c) if rng.random() < ratio else next(ru_c)

    return pick


def move_to_device(example: dict, device: torch.device, dtype: torch.dtype) -> dict:
    out = {}
    for key, value in example.items():
        if key == "input_features":
            out[key] = value.to(device=device, dtype=dtype)
        elif key == "label_weights":
            out[key] = value.to(device=device, dtype=torch.float32)
        else:
            out[key] = value.to(device=device)
    return out


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=dtype)
    import contextlib

    return contextlib.nullcontext()


def build_microbatch(
    clips: list[ClipRef],
    processor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    loss_weights: LossWeightConfig | None = None,
) -> dict[str, torch.Tensor]:
    examples = [
        build_training_example(clip, processor, loss_weights=loss_weights) for clip in clips
    ]
    batch = collate_training_examples(examples)
    return move_to_device(batch, device, dtype)


def compute_total_loss(
    model,
    example: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = example["labels"]
    lw = example["label_weights"]
    model_inputs = {k: v for k, v in example.items() if k not in _META_KEYS}
    out = model(**model_inputs)
    ce = weighted_lm_loss(out.logits, labels, lw)
    return ce, {"ce": float(ce.detach().item()), "total": float(ce.detach().item())}


@torch.no_grad()
def evaluate_dev_loss(
    model,
    processor,
    dev_clips: list[ClipRef],
    device,
    dtype,
    *,
    max_clips: int | None = 32,
    loss_weights: LossWeightConfig | None = None,
) -> float:
    model.eval()
    clips = dev_clips if max_clips is None else dev_clips[: max(1, max_clips)]
    total_loss, n = 0.0, 0
    for clip in tqdm(clips, desc="dev-loss", leave=False, unit="clip"):
        example = move_to_device(
            build_training_example(clip, processor, loss_weights=loss_weights),
            device,
            dtype,
        )
        with _autocast_context(device, dtype):
            loss, _ = compute_total_loss(model, example)
        total_loss += float(loss.item())
        n += 1
    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_moss_metrics(
    model,
    processor,
    clips: list[ClipRef],
    device,
    dtype,
    *,
    max_clips: int = 8,
    max_new_tokens: int = 512,
    en_max_clips: int = 8,
) -> dict[str, float]:
    """CER/cpCER/Δcp on diar clips + separate EN CER for retention."""
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
    )

    model.eval()

    def _gen_score(pool: list[ClipRef], desc: str) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for clip in tqdm(pool, desc=desc, leave=False, unit="clip"):
            messages = build_transcription_messages(clip.audio_path)
            result = generate_transcription(
                model,
                processor,
                messages,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            out.append(score_clip_hypothesis(clip, result["text"]))
        return out

    diar = [c for c in clips if c.target and c.target.count("[S") >= 2]
    pool = (diar if diar else list(clips))[: max(1, max_clips)]
    scores = _gen_score(pool, "dev-metrics")
    agg = aggregate_clip_scores(scores)

    en_pool = [c for c in clips if clip_language(c) == "en"][: max(0, en_max_clips)]
    if en_pool:
        en_scores = _gen_score(en_pool, "en-metrics")
        en_agg = aggregate_clip_scores(en_scores)
        agg["en_cer"] = en_agg["cer"]
        agg["en_n"] = en_agg["n"]
    model.train()
    return agg


def train(
    model,
    processor,
    train_clips: list[ClipRef],
    dev_clips: list[ClipRef],
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    batch_size: int = 1,
    batch_accum: int = 8,
    lr: float = 1e-4,
    log_every: int = 10,
    eval_every: int = 100,
    eval_max_clips: int = 32,
    metrics_every: int | None = None,
    metrics_max_clips: int = 8,
    metrics_max_new_tokens: int = 512,
    en_sample_ratio: float = 0.25,
    loss_weights: LossWeightConfig | None = None,
    on_log=None,
) -> list[dict]:
    """Train with weighted CE only (speaker / timestamp / text)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_accum < 1:
        raise ValueError("batch_accum must be >= 1")

    loss_weights = loss_weights or LossWeightConfig()
    metrics_every = eval_every if metrics_every is None else metrics_every

    model.to(device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    history: list[dict] = []
    running_loss = 0.0
    window_loss = 0.0
    window_n = 0
    last_dev: float | None = None
    last_metrics: dict[str, float] | None = None
    optimizer.zero_grad()
    t0 = time.perf_counter()

    n_en = sum(1 for c in train_clips if clip_language(c) == "en")
    pick = make_en_retention_picker(train_clips, en_ratio=en_sample_ratio)
    pbar = tqdm(
        range(1, steps + 1),
        desc="train",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.3,
    )
    for step in pbar:
        micro = [pick() for _ in range(batch_size)]
        example = build_microbatch(
            micro, processor, device, dtype, loss_weights=loss_weights
        )

        with _autocast_context(device, dtype):
            loss, parts = compute_total_loss(model, example)
            loss = loss / batch_accum

        loss.backward()
        step_loss = float(loss.item()) * batch_accum
        running_loss += step_loss
        window_loss += step_loss
        window_n += 1

        if step % batch_accum == 0:
            optimizer.step()
            optimizer.zero_grad()

        postfix: dict[str, object] = {
            "loss": f"{window_loss / max(window_n, 1):.3f}",
            "eff": batch_size * batch_accum,
            "en%": f"{100 * en_sample_ratio:.0f}" if n_en else "0",
        }
        if last_dev is not None:
            postfix["dev"] = f"{last_dev:.3f}"
        if last_metrics is not None:
            postfix["cer"] = f"{last_metrics['cer']:.3f}"
            postfix["cp"] = f"{last_metrics['cp_cer']:.3f}"
            if "en_cer" in last_metrics:
                postfix["en"] = f"{last_metrics['en_cer']:.3f}"
        pbar.set_postfix(postfix, refresh=False)

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            entry: dict = {
                "step": step,
                "train_loss": avg_loss,
                "batch_size": batch_size,
                "batch_accum": batch_accum,
                "eff_batch": batch_size * batch_accum,
                "elapsed_sec": time.perf_counter() - t0,
                "en_sample_ratio": en_sample_ratio,
                "n_en_train": n_en,
            }
            running_loss = 0.0
            window_loss = 0.0
            window_n = 0

            if step % eval_every == 0 and dev_clips:
                last_dev = evaluate_dev_loss(
                    model,
                    processor,
                    dev_clips,
                    device,
                    dtype,
                    max_clips=eval_max_clips,
                    loss_weights=loss_weights,
                )
                entry["dev_loss"] = last_dev
                entry["dev_eval_clips"] = min(len(dev_clips), eval_max_clips)
                tqdm.write(
                    f"eval step={step} train_loss={avg_loss:.4f} "
                    f"dev_loss={last_dev:.4f} (n={entry['dev_eval_clips']})"
                )

            if metrics_every > 0 and step % metrics_every == 0 and dev_clips:
                last_metrics = evaluate_moss_metrics(
                    model,
                    processor,
                    dev_clips,
                    device,
                    dtype,
                    max_clips=metrics_max_clips,
                    max_new_tokens=metrics_max_new_tokens,
                )
                entry.update(
                    {
                        "cer": last_metrics["cer"],
                        "cp_cer": last_metrics["cp_cer"],
                        "delta_cp": last_metrics["delta_cp"],
                        "metrics_n": last_metrics["n"],
                    }
                )
                msg = (
                    f"metrics step={step} CER={last_metrics['cer']:.4f} "
                    f"cpCER={last_metrics['cp_cer']:.4f} "
                    f"Δcp={last_metrics['delta_cp']:.4f} (n={last_metrics['n']})"
                )
                if "en_cer" in last_metrics:
                    entry["en_cer"] = last_metrics["en_cer"]
                    entry["en_n"] = last_metrics["en_n"]
                    msg += f" EN_CER={last_metrics['en_cer']:.4f} (n={last_metrics['en_n']})"
                tqdm.write(msg)

            history.append(entry)
            if on_log:
                on_log(entry)

    return history

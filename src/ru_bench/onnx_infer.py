"""ONNX KV-cache ASR runtime: sessions, prefill, greedy/beam decode."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ru_bench.onnx_kv import N_LAYERS, past_names, present_names


def strip_gen_specials(text: str) -> str:
    """Remove decode artifacts that break ``parse_transcript`` (drops last turn)."""
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    return text.strip()


def cpu_only(providers: list[str] | None) -> bool:
    if not providers:
        return False
    names: list[str] = []
    for p in providers:
        if isinstance(p, (tuple, list)):
            names.append(str(p[0]))
        else:
            names.append(str(p))
    return all(
        n == "CPUExecutionProvider"
        or n.startswith("CPU")
        or n == "OpenVINOExecutionProvider"
        for n in names
    )


def _ensure_openvino_win_libs(providers: list | None) -> None:
    names = []
    for p in providers or []:
        names.append(p[0] if isinstance(p, (tuple, list)) else str(p))
    if "OpenVINOExecutionProvider" not in names:
        return
    try:
        import onnxruntime.tools.add_openvino_win_libs as ov_utils

        ov_utils.add_openvino_libs_to_path()
    except Exception as exc:  # noqa: BLE001
        print(f"openvino win libs skipped: {exc}", flush=True)


def normalize_providers(providers: list[str] | None) -> list:
    """Expand CLI names; attach OpenVINO CPU device options."""
    if not providers:
        return list(ort.get_available_providers())
    out: list = []
    for p in providers:
        name = p[0] if isinstance(p, (tuple, list)) else str(p)
        if name in ("OpenVINOExecutionProvider", "OpenVINO"):
            opts = p[1] if isinstance(p, (tuple, list)) and len(p) > 1 else {}
            merged = {"device_type": "CPU", **dict(opts)}
            out.append(("OpenVINOExecutionProvider", merged))
        else:
            out.append(p if isinstance(p, (tuple, list)) else name)
    return out


def make_session(
    onnx_path: Path,
    providers: list[str] | None,
    *,
    intra_op_threads: int | None = None,
    inter_op_threads: int = 1,
) -> ort.InferenceSession:
    providers = normalize_providers(providers)
    _ensure_openvino_win_libs(providers)
    if not cpu_only(providers):
        try:
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.is_dir():
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        except Exception as exc:  # noqa: BLE001
            print(f"torch lib PATH skipped: {exc}", flush=True)
        try:
            ort.preload_dlls(cuda=True, cudnn=True)
        except Exception as exc:  # noqa: BLE001
            print(f"preload_dlls skipped: {exc}", flush=True)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    n_threads = intra_op_threads if intra_op_threads is not None else min(8, os.cpu_count() or 4)
    so.intra_op_num_threads = max(1, n_threads)
    so.inter_op_num_threads = max(1, inter_op_threads)
    try:
        from onnxruntime_extensions import get_library_path

        so.register_custom_ops_library(get_library_path())
    except Exception as exc:  # noqa: BLE001
        print(f"extensions not loaded: {exc}", flush=True)

    print(
        f"session {onnx_path.name}: providers={providers} "
        f"intra_op={so.intra_op_num_threads} inter_op={so.inter_op_num_threads} "
        f"ort={ort.__version__}",
        flush=True,
    )
    return ort.InferenceSession(str(onnx_path), so, providers=providers)


def scatter_audio_np(
    token_embeds: np.ndarray,
    input_ids: np.ndarray,
    audio_embeds: np.ndarray,
    audio_token_id: int,
) -> np.ndarray:
    out = token_embeds.copy()
    flat_audio = audio_embeds.reshape(-1, audio_embeds.shape[-1])
    mask = input_ids.reshape(-1) == audio_token_id
    n = int(mask.sum())
    if n != flat_audio.shape[0]:
        raise ValueError(
            f"audio token count {n} != audio_embeds rows {flat_audio.shape[0]}"
        )
    out.reshape(-1, out.shape[-1])[mask] = flat_audio
    return out


def decode_uses_input_ids(session: ort.InferenceSession) -> bool:
    return any(i.name == "input_ids" for i in session.get_inputs())


def numpy_to_ortvalue(arr: np.ndarray) -> ort.OrtValue:
    return ort.OrtValue.ortvalue_from_numpy(arr)


def decode_step_iobinding(
    session: ort.InferenceSession,
    *,
    fused_input_ids: bool,
    token_input: np.ndarray,
    attention_mask: np.ndarray,
    past_ovs: list[ort.OrtValue],
) -> tuple[np.ndarray, list[ort.OrtValue]]:
    """One decode step; ``past_ovs`` / present stay as OrtValue on CPU."""
    io = session.io_binding()
    if fused_input_ids:
        io.bind_cpu_input("input_ids", token_input)
    else:
        io.bind_cpu_input("inputs_embeds", token_input)
    io.bind_cpu_input("attention_mask", attention_mask)
    for name, ov in zip(past_names(), past_ovs, strict=True):
        io.bind_ortvalue_input(name, ov)
    io.bind_output("logits", "cpu")
    for name in present_names():
        io.bind_output(name, "cpu")
    session.run_with_iobinding(io)
    outs = io.get_outputs()
    logits = outs[0].numpy()
    return logits, list(outs[1:])


def decode_step_batched(
    session: ort.InferenceSession,
    *,
    fused_input_ids: bool,
    token_input: np.ndarray,
    attention_mask: np.ndarray,
    past_np: list[np.ndarray],
    sess_embed: ort.InferenceSession | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Batched decode (beam). ``past_np[i]`` shape ``[B, 8, T, 128]``."""
    feeds: dict[str, np.ndarray] = {"attention_mask": attention_mask.astype(np.int64, copy=False)}
    if fused_input_ids:
        feeds["input_ids"] = token_input.astype(np.int64, copy=False)
    else:
        assert sess_embed is not None
        embeds = [
            sess_embed.run(
                ["inputs_embeds"],
                {"input_ids": row.reshape(1, -1).astype(np.int64)},
            )[0][0]
            for row in token_input
        ]
        feeds["inputs_embeds"] = np.stack(embeds, axis=0).astype(np.float32)
    for name, arr in zip(past_names(), past_np, strict=True):
        feeds[name] = arr
    outs = session.run(None, feeds)
    return outs[0], list(outs[1:])


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64)
    x64 = x64 - np.max(x64, axis=-1, keepdims=True)
    return x64 - np.log(np.exp(x64).sum(axis=-1, keepdims=True))


def _prefill_kv(
    *,
    sess_audio: ort.InferenceSession,
    sess_embed: ort.InferenceSession,
    sess_prefill: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    input_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    audio_token_id: int,
) -> tuple[np.ndarray, list[np.ndarray], float, float]:
    t0 = time.perf_counter()
    audio_embeds = sess_audio.run(
        ["audio_embeds"],
        {
            "input_features": input_features.astype(np.float32),
            "audio_feature_lengths": audio_feature_lengths.astype(np.int64),
        },
    )[0]
    t_audio = time.perf_counter() - t0
    tok_embeds = sess_embed.run(
        ["inputs_embeds"], {"input_ids": input_ids.astype(np.int64)}
    )[0]
    pref_embeds = scatter_audio_np(
        tok_embeds.astype(np.float32),
        input_ids.astype(np.int64),
        audio_embeds.astype(np.float32),
        audio_token_id,
    )
    t_pref = time.perf_counter()
    outs = sess_prefill.run(
        None,
        {
            "inputs_embeds": pref_embeds,
            "attention_mask": attention_mask.astype(np.int64),
        },
    )
    past_np = [outs[i + 1] for i in range(N_LAYERS * 2)]
    return outs[0], past_np, t_audio, time.perf_counter() - t_pref


def greedy_decode_kv(
    *,
    sess_audio: ort.InferenceSession,
    sess_embed: ort.InferenceSession,
    sess_prefill: ort.InferenceSession,
    sess_decode: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    input_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    audio_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    log_every: int = 10,
) -> np.ndarray:
    t0 = time.perf_counter()
    fused = decode_uses_input_ids(sess_decode)
    print(f"decode mode={'fused-input_ids' if fused else 'embeds+IOBinding'}", flush=True)

    logits, past_np, t_audio, t_pref = _prefill_kv(
        sess_audio=sess_audio,
        sess_embed=sess_embed,
        sess_prefill=sess_prefill,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        audio_feature_lengths=audio_feature_lengths,
        audio_token_id=audio_token_id,
    )
    print(f"audio embeds in {t_audio:.2f}s", flush=True)
    print(f"prefill done in {t_pref:.2f}s", flush=True)
    past_ovs = [numpy_to_ortvalue(p) for p in past_np]

    ids = input_ids.copy()
    mask = attention_mask.astype(np.int64).copy()
    next_id = int(np.argmax(logits[0, -1]))
    ids = np.concatenate([ids, np.array([[next_id]], dtype=ids.dtype)], axis=1)
    mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)

    t_dec = time.perf_counter()
    for step in range(1, max_new_tokens):
        if next_id == eos_token_id:
            print(f"decode EOS at step={step}", flush=True)
            break

        if fused:
            token_in = np.array([[next_id]], dtype=np.int64)
        else:
            token_in = sess_embed.run(
                ["inputs_embeds"],
                {"input_ids": np.array([[next_id]], dtype=np.int64)},
            )[0].astype(np.float32)

        logits, past_ovs = decode_step_iobinding(
            sess_decode,
            fused_input_ids=fused,
            token_input=token_in,
            attention_mask=mask,
            past_ovs=past_ovs,
        )
        next_id = int(np.argmax(logits[0, -1]))
        ids = np.concatenate([ids, np.array([[next_id]], dtype=ids.dtype)], axis=1)
        mask = np.concatenate([mask, np.ones((1, 1), dtype=np.int64)], axis=1)

        if log_every > 0 and step % log_every == 0:
            elapsed = time.perf_counter() - t_dec
            print(
                f"decode step={step} tok/s={step / max(elapsed, 1e-6):.2f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(f"total {time.perf_counter() - t0:.1f}s", flush=True)
    return ids


def beam_decode_kv(
    *,
    sess_audio: ort.InferenceSession,
    sess_embed: ort.InferenceSession,
    sess_prefill: ort.InferenceSession,
    sess_decode: ort.InferenceSession,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    input_features: np.ndarray,
    audio_feature_lengths: np.ndarray,
    audio_token_id: int,
    eos_token_id: int,
    max_new_tokens: int,
    beam_size: int = 3,
    log_every: int = 10,
) -> np.ndarray:
    """Length-normalized beam search; ORT batch = open beams only."""
    if beam_size < 2:
        return greedy_decode_kv(
            sess_audio=sess_audio,
            sess_embed=sess_embed,
            sess_prefill=sess_prefill,
            sess_decode=sess_decode,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            audio_feature_lengths=audio_feature_lengths,
            audio_token_id=audio_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            log_every=log_every,
        )

    t0 = time.perf_counter()
    fused = decode_uses_input_ids(sess_decode)
    B = int(beam_size)
    print(f"decode mode=beam{B}-active-batch fused={fused}", flush=True)

    logits, past_np, t_audio, t_pref = _prefill_kv(
        sess_audio=sess_audio,
        sess_embed=sess_embed,
        sess_prefill=sess_prefill,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        audio_feature_lengths=audio_feature_lengths,
        audio_token_id=audio_token_id,
    )
    print(f"audio embeds in {t_audio:.2f}s", flush=True)
    print(f"prefill done in {t_pref:.2f}s", flush=True)

    def _norm(score: float, n_tok: int) -> float:
        return score / max(n_tok, 1)

    lp0 = _log_softmax(logits[0, -1])
    top0 = np.argpartition(lp0, -B)[-B:]
    top0 = top0[np.argsort(-lp0[top0])]

    finished: list[tuple[float, list[int]]] = []
    open_scores = lp0[top0].astype(np.float64).copy()
    open_toks: list[list[int]] = [[int(t)] for t in top0]
    still: list[int] = []
    for i, t in enumerate(top0):
        if int(t) == eos_token_id:
            finished.append((float(open_scores[i]), [int(t)]))
        else:
            still.append(i)
    if not still and finished:
        best_s, best_t = max(finished, key=lambda x: _norm(x[0], len(x[1])))
        out = np.concatenate([input_ids[0], np.array(best_t, dtype=np.int64)], axis=0)[None, :]
        print(f"total {time.perf_counter() - t0:.1f}s (beam={B}, all EOS@0)", flush=True)
        return out

    open_scores = open_scores[still]
    open_toks = [open_toks[i] for i in still]
    first_tok = np.array([[open_toks[i][0]] for i in range(len(still))], dtype=np.int64)
    A = len(still)
    past_a = [np.repeat(p, A, axis=0) for p in past_np]
    mask_a = np.repeat(attention_mask.astype(np.int64), A, axis=0)
    mask_a = np.concatenate([mask_a, np.ones((A, 1), dtype=np.int64)], axis=1)

    t_dec = time.perf_counter()
    n_ort = 0
    n_ort_rows = 0

    def _decode_open(token_ba: np.ndarray) -> np.ndarray:
        nonlocal past_a, mask_a, n_ort, n_ort_rows
        a = token_ba.shape[0]
        logits_a, past_a = decode_step_batched(
            sess_decode,
            fused_input_ids=fused,
            token_input=token_ba,
            attention_mask=mask_a,
            past_np=past_a,
            sess_embed=None if fused else sess_embed,
        )
        n_ort += 1
        n_ort_rows += a
        return logits_a

    logits_a = _decode_open(first_tok)

    for step in range(1, max_new_tokens):
        if A == 0:
            print(f"beam all EOS at step={step}", flush=True)
            break

        cand: list[tuple[float, list[int], int | None, int]] = []
        for fi, (fs, ft) in enumerate(finished):
            cand.append((fs, ft, None, -1))
        for b in range(A):
            lp = _log_softmax(logits_a[b, -1])
            top = np.argpartition(lp, -B)[-B:]
            for tid in top:
                tid_i = int(tid)
                cand.append(
                    (
                        float(open_scores[b]) + float(lp[tid_i]),
                        open_toks[b] + [tid_i],
                        b,
                        tid_i,
                    )
                )

        cand.sort(key=lambda c: _norm(c[0], len(c[1])), reverse=True)
        cand = cand[:B]

        new_finished: list[tuple[float, list[int]]] = []
        new_scores: list[float] = []
        new_toks: list[list[int]] = []
        parent_open: list[int] = []
        next_toks: list[int] = []
        for score, toks, parent, ntok in cand:
            if parent is None:
                new_finished.append((score, toks))
                continue
            if ntok == eos_token_id:
                new_finished.append((score, toks))
                continue
            new_scores.append(score)
            new_toks.append(toks)
            parent_open.append(parent)
            next_toks.append(ntok)

        finished = new_finished
        A = len(new_scores)
        if A == 0:
            break

        pidx = np.array(parent_open, dtype=np.int64)
        past_a = [p[pidx] for p in past_a]
        mask_a = mask_a[pidx]
        mask_a = np.concatenate([mask_a, np.ones((A, 1), dtype=np.int64)], axis=1)
        open_scores = np.array(new_scores, dtype=np.float64)
        open_toks = new_toks
        token_ba = np.array([[t] for t in next_toks], dtype=np.int64)
        logits_a = _decode_open(token_ba)

        if log_every > 0 and step % log_every == 0:
            elapsed = time.perf_counter() - t_dec
            print(
                f"beam step={step} open={A}/{B} fin={len(finished)} "
                f"ort_runs={n_ort} ort_rows={n_ort_rows} elapsed={elapsed:.1f}s",
                flush=True,
            )

    pool = finished + [(float(open_scores[i]), open_toks[i]) for i in range(A)]
    if not pool:
        raise RuntimeError("beam search produced empty hypothesis pool")
    best_s, best_t = max(pool, key=lambda x: _norm(x[0], len(x[1])))
    gen = np.array(best_t, dtype=np.int64)
    out = np.concatenate([input_ids[0], gen], axis=0)[None, :]
    elapsed = time.perf_counter() - t0
    print(
        f"total {elapsed:.1f}s (beam={B}, ort_runs={n_ort}, ort_rows={n_ort_rows}, "
        f"rows/run~{n_ort_rows / max(n_ort, 1):.2f})",
        flush=True,
    )
    return out


def graph_path(model_dir: Path, name: str, quant: str) -> Path:
    if quant == "fp32":
        return model_dir / f"{name}.onnx"
    path = model_dir / f"{name}.{quant}.onnx"
    if path.is_file():
        return path
    print(f"missing {path}, fallback {name}.onnx", flush=True)
    return model_dir / f"{name}.onnx"


def resolve_kv_paths(
    model_dir: Path,
    *,
    quant: str = "fp32",
    trim_mel: bool = False,
    audio_onnx: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Return (audio, prefill, decode, embed) ONNX paths for a KV pack."""
    if quant == "cpu-fast":
        audio = graph_path(model_dir, "audio", "fp32")
        prefill = graph_path(model_dir, "lm_prefill", "fp32")
        opt_dec = model_dir / "lm_decode.opt.dynint8.onnx"
        decode = opt_dec if opt_dec.is_file() else graph_path(model_dir, "lm_decode", "dynint8")
    else:
        audio = graph_path(model_dir, "audio", quant)
        prefill = graph_path(model_dir, "lm_prefill", quant)
        decode = graph_path(model_dir, "lm_decode", quant)

    dynpos = model_dir / "audio.dynpos.onnx"
    if audio_onnx is not None:
        audio = audio_onnx
    elif trim_mel and dynpos.is_file():
        audio = dynpos
    elif trim_mel:
        print(
            "warn: --trim-mel set but audio.dynpos.onnx missing — "
            "run scripts/20_patch_audio_dynpos.py",
            flush=True,
        )

    return audio, prefill, decode, model_dir / "embed.onnx"

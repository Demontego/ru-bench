"""ru-bench unified CLI.

  uv run ru-bench bench download
  uv run ru-bench data manifests --sources mixed
  uv run ru-bench train finetune --steps 1000
  uv run ru-bench eval golos --checkpoint-dir checkpoints/lora_ru
  uv run ru-bench onnx export-kv --checkpoint-dir checkpoints/lora_ru_fmt3
  uv run ru-bench onnx infer --model-dir exports/moss_ru_fmt3_kv --audio file.wav
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return
    if argv[0] in {"-V", "--version"}:
        from importlib.metadata import version

        print(version("ru-bench"))
        return

    group, rest = argv[0], argv[1:]
    if group == "bench":
        from ru_bench.cli.bench import dispatch

        dispatch(rest)
        return

    if len(rest) < 1 or rest[0] in {"-h", "--help"}:
        _print_group_help(group)
        return

    action, action_argv = rest[0], rest[1:]
    _run_action(group, action, action_argv)


def _run_action(group: str, action: str, action_argv: list[str]) -> None:
    old = sys.argv
    sys.argv = [old[0], *action_argv]
    try:
        fn = _ACTIONS.get((group, action))
        if fn is None:
            raise SystemExit(f"unknown command: {group} {action}\n{_group_actions(group)}")
        fn()
    finally:
        sys.argv = old


def _lazy_actions() -> dict[tuple[str, str], object]:
    from ru_bench.cli.data_diar_sample import main as data_diar
    from ru_bench.cli.data_error_mine import main as data_mine
    from ru_bench.cli.data_inject_en import main as data_inject
    from ru_bench.cli.data_manifests import main as data_manifests
    from ru_bench.cli.eval_diar import main as eval_diar
    from ru_bench.cli.eval_golos import main as eval_golos
    from ru_bench.cli.eval_onnx_trim import main as eval_trim
    from ru_bench.cli.eval_retention import main as eval_retention
    from ru_bench.cli.infer_kv import main as onnx_infer
    from ru_bench.cli.onnx_export import main as onnx_export
    from ru_bench.cli.onnx_install_ort import main as onnx_install
    from ru_bench.cli.onnx_patch_audio import main as onnx_patch
    from ru_bench.cli.train_finetune import main as train_finetune
    from ru_bench.cli.train_vram import main as train_vram

    return {
        ("data", "manifests"): data_manifests,
        ("data", "inject-en"): data_inject,
        ("data", "diar-sample"): data_diar,
        ("data", "error-mine"): data_mine,
        ("train", "finetune"): train_finetune,
        ("train", "vram-probe"): train_vram,
        ("eval", "golos"): eval_golos,
        ("eval", "retention"): eval_retention,
        ("eval", "diar"): eval_diar,
        ("eval", "onnx-trim"): eval_trim,
        ("onnx", "export-kv"): onnx_export,
        ("onnx", "patch-audio"): onnx_patch,
        ("onnx", "install-ort"): onnx_install,
        ("onnx", "infer"): onnx_infer,
    }


_ACTIONS: dict[tuple[str, str], object] = {}


def _ensure_actions() -> None:
    global _ACTIONS
    if not _ACTIONS:
        _ACTIONS = _lazy_actions()


def _group_actions(group: str) -> str:
    _ensure_actions()
    lines = [f"  {a}" for (g, a) in sorted(_ACTIONS) if g == group]
    return "\n".join(lines) or "  (none)"


def _print_group_help(group: str) -> None:
    print(f"ru-bench {group} <action> [options]\n")
    print(_group_actions(group))


def _print_help() -> None:
    print(
        """ru-bench — MOSS Russian ASR benchmark + LoRA + ONNX

Groups:
  bench   download | infer | metrics | report
  data    manifests | inject-en | diar-sample | error-mine
  train   finetune | vram-probe
  eval    golos | retention | diar | onnx-trim
  onnx    export-kv | patch-audio | install-ort | infer

Examples:
  uv run ru-bench bench download --asr-n 100
  uv run ru-bench bench infer
  uv run ru-bench onnx infer --model-dir exports/moss_ru_fmt3_kv --audio clip.wav

See docs/COMMANDS.md"""
    )


if __name__ == "__main__":
    main()

"""ru-bench unified CLI.

  uv run ru-bench bench download
  uv run ru-bench data manifests --sources mixed
  uv run ru-bench train finetune --steps 1000
  uv run ru-bench eval golos --checkpoint-dir checkpoints/lora_ru
  uv run ru-bench onnx export-kv --checkpoint-dir checkpoints/lora_ru_fmt3
  uv run ru-bench onnx infer --model-dir exports/moss_ru_fmt3_kv --audio file.wav
"""

from __future__ import annotations

import importlib
import sys

# Static help — no heavy imports (onnx/torch load only on dispatch).
_GROUP_ACTIONS: dict[str, list[str]] = {
    "data": ["manifests", "inject-en", "diar-sample", "error-mine"],
    "train": ["finetune", "vram-probe"],
    "eval": ["golos", "retention", "diar", "onnx-trim"],
    "onnx": ["export-kv", "patch-audio", "install-ort", "infer"],
}

# module path → main callable
_ACTION_MODULES: dict[tuple[str, str], str] = {
    ("data", "manifests"): "ru_bench.cli.data_manifests",
    ("data", "inject-en"): "ru_bench.cli.data_inject_en",
    ("data", "diar-sample"): "ru_bench.cli.data_diar_sample",
    ("data", "error-mine"): "ru_bench.cli.data_error_mine",
    ("train", "finetune"): "ru_bench.cli.train_finetune",
    ("train", "vram-probe"): "ru_bench.cli.train_vram",
    ("eval", "golos"): "ru_bench.cli.eval_golos",
    ("eval", "retention"): "ru_bench.cli.eval_retention",
    ("eval", "diar"): "ru_bench.cli.eval_diar",
    ("eval", "onnx-trim"): "ru_bench.cli.eval_onnx_trim",
    ("onnx", "export-kv"): "ru_bench.cli.onnx_export",
    ("onnx", "patch-audio"): "ru_bench.cli.onnx_patch_audio",
    ("onnx", "install-ort"): "ru_bench.cli.onnx_install_ort",
    ("onnx", "infer"): "ru_bench.cli.infer_kv",
}


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

    if group not in _GROUP_ACTIONS:
        raise SystemExit(f"unknown group: {group}\n\n{_print_groups()}")

    if len(rest) < 1 or rest[0] in {"-h", "--help"}:
        _print_group_help(group)
        return

    action, action_argv = rest[0], rest[1:]
    _run_action(group, action, action_argv)


def _run_action(group: str, action: str, action_argv: list[str]) -> None:
    mod_path = _ACTION_MODULES.get((group, action))
    if mod_path is None:
        raise SystemExit(f"unknown command: {group} {action}\n{_group_actions(group)}")

    mod = importlib.import_module(mod_path)
    main_fn = getattr(mod, "main")

    old = sys.argv
    sys.argv = [old[0], *action_argv]
    try:
        main_fn()
    finally:
        sys.argv = old


def _group_actions(group: str) -> str:
    actions = _GROUP_ACTIONS.get(group, [])
    return "\n".join(f"  {a}" for a in actions) or "  (none)"


def _print_groups() -> str:
    return "Groups: " + ", ".join(sorted(_GROUP_ACTIONS))


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
  uv sync --extra onnx-cpu
  uv run ru-bench onnx infer --model-dir exports/moss_ru_fmt3_kv --audio clip.wav

See docs/COMMANDS.md"""
    )


if __name__ == "__main__":
    main()

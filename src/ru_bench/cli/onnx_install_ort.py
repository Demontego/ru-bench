"""Install onnxruntime from GitHub (tag) when PyPI wheels lag.

Two paths:

1) **Native DLL swap** (Windows, fast) — download release zip, replace
   ``site-packages/onnxruntime/capi/*.dll``. Python package metadata may still
   print 1.27.0; native binary is 1.27.1.

     uv run python scripts/19_install_ort_from_git.py --tag v1.27.1 --dll-swap

2) **Full source wheel** (needs VS Build Tools + CMake) — long:

     uv run python scripts/19_install_ort_from_git.py --tag v1.27.1 --build --install
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def _download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 1_000_000:
        print(f"reuse {out} ({out.stat().st_size} bytes)", flush=True)
        return out
    print(f"download {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ru-bench"})
    with urllib.request.urlopen(req, timeout=600) as resp, out.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"saved {out} ({out.stat().st_size} bytes)", flush=True)
    return out


def dll_swap(tag: str, deps: Path) -> None:
    """Replace venv ORT DLLs from GitHub native win-x64 package."""
    ver = tag.lstrip("v")
    zip_name = f"onnxruntime-win-x64-{ver}.zip"
    url = f"https://github.com/microsoft/onnxruntime/releases/download/{tag}/{zip_name}"
    zpath = _download(url, deps / zip_name)
    extract = deps / f"ort_native_{ver}"
    if extract.exists():
        shutil.rmtree(extract)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract)
    libs = list(extract.rglob("onnxruntime.dll"))
    if not libs:
        raise SystemExit(f"onnxruntime.dll not found under {extract}")
    src_lib = libs[0].parent

    # Resolve capi without importing onnxruntime (DLL lock on Windows).
    import site

    candidates: list[Path] = []
    venv = Path(sys.prefix) / "Lib" / "site-packages" / "onnxruntime" / "capi"
    candidates.append(venv)
    for sp in site.getsitepackages():
        candidates.append(Path(sp) / "onnxruntime" / "capi")
    user_sp = site.getusersitepackages()
    if user_sp:
        candidates.append(Path(user_sp) / "onnxruntime" / "capi")
    capi = next((c for c in candidates if (c / "onnxruntime.dll").is_file()), None)
    if capi is None:
        raise SystemExit("onnxruntime capi not found in site-packages")

    for name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
        src = src_lib / name
        dst = capi / name
        if not src.is_file():
            print(f"skip missing {src}", flush=True)
            continue
        bak = dst.with_suffix(dst.suffix + f".bak_pre_{ver}")
        if dst.is_file() and not bak.is_file():
            shutil.copy2(dst, bak)
        shutil.copy2(src, dst)
        print(f"replaced {dst} <- {src} ({src.stat().st_size} bytes)", flush=True)

    print(
        "DLL swap done. Python metadata may still show older version; "
        "native binary is from the release zip.",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="v1.27.1")
    p.add_argument("--dir", type=Path, default=Path("deps/onnxruntime"))
    p.add_argument("--deps", type=Path, default=Path("deps"))
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    p.add_argument(
        "--dll-swap",
        action="store_true",
        help="Windows: swap native DLLs from release zip",
    )
    p.add_argument("--build", action="store_true", help="Build wheel from cloned sources")
    p.add_argument("--skip-clone", action="store_true")
    p.add_argument("--install", action="store_true", help="uv pip install built wheel")
    args = p.parse_args()

    root = Path.cwd()
    deps = args.deps if args.deps.is_absolute() else root / args.deps
    dest = args.dir if args.dir.is_absolute() else root / args.dir

    if args.dll_swap:
        dll_swap(args.tag, deps)
        return

    if not args.build and not args.install:
        raise SystemExit("pass --dll-swap (recommended on Windows) or --build")

    if args.build and not args.skip_clone:
        if dest.is_dir() and (dest / ".git").is_dir():
            print(f"fetch/checkout {args.tag} in {dest}", flush=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(dest),
                    "fetch",
                    "--tags",
                    "--depth",
                    "1",
                    "origin",
                    args.tag,
                ],
                check=False,
            )
            subprocess.run(["git", "-C", str(dest), "checkout", args.tag], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(dest),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--depth",
                    "1",
                ],
                check=False,
            )
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.rmtree(dest)
            print(f"clone {args.tag} → {dest}", flush=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    args.tag,
                    "--recurse-submodules",
                    "https://github.com/microsoft/onnxruntime.git",
                    str(dest),
                ],
                check=True,
            )

    if args.build:
        build_bat = dest / "build.bat"
        if not build_bat.is_file():
            raise SystemExit(f"missing {build_bat} (need VS Build Tools)")
        cmd = [
            str(build_bat),
            "--config",
            "Release",
            "--build_shared_lib",
            "--parallel",
            str(args.jobs),
            "--build_wheel",
            "--skip_tests",
            "--compile_no_warning_as_error",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(dest), check=True)

    if args.install:
        wheels = sorted(
            dest.rglob("onnxruntime-*.whl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if not wheels:
            raise SystemExit("no wheel found — run with --build first")
        wheel = wheels[0]
        print(f"wheel: {wheel}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "--force-reinstall", str(wheel)],
            check=True,
        )


if __name__ == "__main__":
    main()

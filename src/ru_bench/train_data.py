import json
import random
import tarfile
import urllib.request
from pathlib import Path

from ru_bench import config
from ru_bench.data import ClipRef


def download_tar(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tar.part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def extract_tar(tar_path: Path, dest_dir: Path) -> Path:
    if dest_dir.exists():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir)
    return dest_dir


def download_train_farfield(raw_dir: Path = config.GOLOS_TRAIN_RAW_DIR) -> Path:
    """Download farfield audio. Manifests live in train_crowd9.tar (Golos docs)."""
    tar_path = raw_dir / "train_farfield.tar"
    download_tar(config.GOLOS_TRAIN_FARFIELD_URL, tar_path)
    return extract_tar(tar_path, raw_dir / "train_farfield")


def download_train_crowd_shard(shard: int, raw_dir: Path = config.GOLOS_TRAIN_RAW_DIR) -> Path:
    url = config.GOLOS_TRAIN_CROWD_SHARD_URL.format(shard=shard)
    tar_path = raw_dir / f"train_crowd{shard}.tar"
    download_tar(url, tar_path)
    return extract_tar(tar_path, raw_dir / f"train_crowd{shard}")


def ensure_golos_train_manifests(raw_dir: Path = config.GOLOS_TRAIN_RAW_DIR) -> Path:
    """Golos puts all train manifests in train_crowd9.tar (~8GB), not in farfield.

    Returns extract dir that contains *.jsonl manifests.
    """
    existing = find_manifests(raw_dir)
    if existing:
        return existing[0].parent
    print(
        "Golos train manifests are in train_crowd9.tar (~8GB), not in farfield. "
        "Downloading..."
    )
    return download_train_crowd_shard(9, raw_dir)


def find_manifests(extract_dir: Path) -> list[Path]:
    return sorted(extract_dir.rglob("*.jsonl"))


def _golos_audio_dirs(raw_dir: Path = config.GOLOS_TRAIN_RAW_DIR) -> dict[str, Path]:
    """Known on-disk roots for Golos train audio (CDN extract layout)."""
    return {
        "farfield": raw_dir / "train_farfield" / "train" / "farfield",
        "crowd": raw_dir / "train_crowd9" / "train" / "crowd",  # only if shard extracted
    }


def _resolve_golos_audio(row: dict, audio_dirs: dict[str, Path]) -> tuple[str, Path] | None:
    rel = str(row.get("audio_filepath", "")).replace("\\", "/")
    name = Path(rel).name
    if rel.startswith("farfield/") or "/farfield/" in rel:
        domain = "farfield"
    elif rel.startswith("crowd/") or "/crowd/" in rel:
        domain = "crowd"
    else:
        return None
    root = audio_dirs.get(domain)
    if root is None or not root.exists():
        return None
    # farfield: flat <id>.wav; crowd shards: crowd/<shard>/<id>.wav
    path = root / name if domain == "farfield" else root.parent / rel
    if not path.exists():
        path = root / Path(rel).name
    if not path.exists():
        return None
    return domain, path


def load_clips_from_manifests(
    extract_dir: Path,
    *,
    manifest_roots: list[Path] | None = None,
    domain_filter: str | None = None,
    limit: int | None = None,
) -> list[ClipRef]:
    """Parse Golos train manifests and attach audio that exists on disk.

    Prefer the full ``manifest.jsonl`` from train_crowd9 (not the 10min/1h subsets).
    ``extract_dir`` is kept for API compat; audio resolved via known CDN layout.
    """
    del extract_dir  # layout is fixed under GOLOS_TRAIN_RAW_DIR
    roots = list(manifest_roots or [config.GOLOS_TRAIN_RAW_DIR])
    manifests: list[Path] = []
    seen_manifests: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in find_manifests(root):
            # Skip tiny subset manifests; full list is train/manifest.jsonl
            if path.name != "manifest.jsonl" and "hours" in path.name:
                continue
            if path.name in {"10min.jsonl", "1hour.jsonl", "10hours.jsonl", "100hours.jsonl"}:
                continue
            resolved = path.resolve()
            if resolved not in seen_manifests:
                seen_manifests.add(resolved)
                manifests.append(path)

    if not manifests:
        # Fallback: any jsonl
        for root in roots:
            for path in find_manifests(root):
                manifests.append(path)
                break

    if not manifests:
        raise FileNotFoundError(
            f"No *.jsonl manifest found under {roots}. Golos train manifests ship in "
            "train_crowd9.tar — call ensure_golos_train_manifests() first."
        )

    audio_dirs = _golos_audio_dirs()
    # Also accept farfield extract if user pointed elsewhere
    ff = config.GOLOS_TRAIN_RAW_DIR / "train_farfield" / "train" / "farfield"
    if ff.exists():
        audio_dirs["farfield"] = ff

    clips = []
    seen_ids = set()
    for manifest_path in manifests:
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                if domain_filter == "farfield" and "farfield/" not in line:
                    continue
                if domain_filter == "crowd" and "crowd/" not in line:
                    continue
                row = json.loads(line)
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                clip_id = str(row["id"])
                if clip_id in seen_ids:
                    continue
                resolved = _resolve_golos_audio(row, audio_dirs)
                if resolved is None:
                    continue
                domain, audio_path = resolved
                if domain_filter and domain != domain_filter:
                    continue
                seen_ids.add(clip_id)
                clips.append(
                    ClipRef(
                        clip_id=clip_id,
                        domain=domain,
                        audio_path=str(audio_path),
                        text=text,
                        duration=float(row["duration"]),
                    )
                )
                if limit is not None and len(clips) >= limit:
                    return clips
    return clips


def clip_language(clip: ClipRef) -> str:
    """Map domain to language bucket (ru|en). Unknown domains default to ru."""
    if clip.domain in config.EN_DOMAINS:
        return "en"
    return "ru"


def filter_existing_audio(clips: list[ClipRef]) -> tuple[list[ClipRef], int]:
    """Drop clips whose audio_path is missing. Returns (kept, missing_count)."""
    kept: list[ClipRef] = []
    missing = 0
    for clip in clips:
        if Path(clip.audio_path).is_file():
            kept.append(clip)
        else:
            missing += 1
    return kept, missing


def select_train_dev_split(
    clips: list[ClipRef], n_train: int, n_dev: int, seed: int = config.SEED
) -> tuple[list[ClipRef], list[ClipRef]]:
    """Legacy random split. Prefer select_train_dev_split_ru_primary when n_train/n_dev are 0."""
    if n_train <= 0 and n_dev <= 0:
        return select_train_dev_split_ru_primary(clips, seed=seed)
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    if n_dev <= 0:
        n_dev = max(1, int(round(len(shuffled) * config.EVAL_RATIO)))
    if n_train <= 0:
        n_train = max(0, len(shuffled) - n_dev)
    dev = shuffled[:n_dev]
    train = shuffled[n_dev : n_dev + n_train]
    return train, dev


def select_train_dev_split_ru_primary(
    clips: list[ClipRef],
    *,
    eval_ratio: float = config.EVAL_RATIO,
    seed: int = config.SEED,
) -> tuple[list[ClipRef], list[ClipRef]]:
    """RU-primary train + ~50/50 eval; uses all clips (no leftover).

    - Eval = ``eval_ratio`` of total pool. Prefer 50/50 RU/EN; if EN short,
      take all available EN up to half of eval, fill rest with RU.
    - Train = everything remaining (RU-heavy OK; modest EN for retention).
    """
    if not clips:
        raise ValueError("Empty clip pool")

    rng = random.Random(seed)
    by_lang: dict[str, list[ClipRef]] = {"ru": [], "en": []}
    for clip in clips:
        by_lang[clip_language(clip)].append(clip)
    for lang in by_lang:
        rng.shuffle(by_lang[lang])

    n_total = len(clips)
    n_dev = max(1, int(round(n_total * eval_ratio))) if eval_ratio > 0 else 0
    n_dev = min(n_dev, n_total - 1) if n_total > 1 else min(n_dev, n_total)

    # Prefer 50/50 eval; cap EN by availability and half of eval.
    n_dev_en = min(len(by_lang["en"]), n_dev // 2)
    n_dev_ru = n_dev - n_dev_en
    if n_dev_ru > len(by_lang["ru"]):
        # Not enough RU — pull extra EN if any remain.
        n_dev_ru = len(by_lang["ru"])
        n_dev_en = min(len(by_lang["en"]), n_dev - n_dev_ru)

    dev = by_lang["ru"][:n_dev_ru] + by_lang["en"][:n_dev_en]
    train = by_lang["ru"][n_dev_ru:] + by_lang["en"][n_dev_en:]

    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


# Back-compat alias (old name meant forced 50/50 everywhere — now RU-primary).
select_train_dev_split_balanced = select_train_dev_split_ru_primary


def save_clip_manifest(clips: list[ClipRef], path: Path) -> None:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in clips], f, ensure_ascii=False, indent=2)

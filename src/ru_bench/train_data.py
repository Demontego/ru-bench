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
    tar_path = raw_dir / "train_farfield.tar"
    download_tar(config.GOLOS_TRAIN_FARFIELD_URL, tar_path)
    return extract_tar(tar_path, raw_dir / "train_farfield")


def download_train_crowd_shard(shard: int, raw_dir: Path = config.GOLOS_TRAIN_RAW_DIR) -> Path:
    url = config.GOLOS_TRAIN_CROWD_SHARD_URL.format(shard=shard)
    tar_path = raw_dir / f"train_crowd{shard}.tar"
    download_tar(url, tar_path)
    return extract_tar(tar_path, raw_dir / f"train_crowd{shard}")


def find_manifests(extract_dir: Path) -> list[Path]:
    return sorted(extract_dir.rglob("*.jsonl"))


def load_clips_from_manifests(extract_dir: Path) -> list[ClipRef]:
    """Parse every manifest.jsonl found under extract_dir, skipping rows whose
    audio file isn't actually present (Golos train archives are split across
    shards; a manifest may reference audio that lives in a shard we didn't
    download).
    """
    manifests = find_manifests(extract_dir)
    if not manifests:
        raise FileNotFoundError(
            f"No *.jsonl manifest found under {extract_dir}. Golos's crowd train shards may only "
            "bundle the full manifest inside train_crowd9.tar (per Golos's own docs) rather than "
            "each shard being self-contained like the test split was. Try downloading shard 9 "
            "(download_train_crowd_shard(9)) alongside whichever shard(s) hold the audio you want."
        )

    clips = []
    seen_ids = set()
    for manifest_path in manifests:
        domain_dir = manifest_path.parent
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if not row["text"].strip():
                    continue
                if row["id"] in seen_ids:
                    continue
                audio_path = domain_dir / row["audio_filepath"]
                if not audio_path.exists():
                    continue
                seen_ids.add(row["id"])
                clips.append(
                    ClipRef(
                        clip_id=row["id"],
                        domain=domain_dir.name,
                        audio_path=str(audio_path),
                        text=row["text"],
                        duration=row["duration"],
                    )
                )
    return clips


def select_train_dev_split(
    clips: list[ClipRef], n_train: int, n_dev: int, seed: int = config.SEED
) -> tuple[list[ClipRef], list[ClipRef]]:
    rng = random.Random(seed)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    dev = shuffled[:n_dev]
    train = shuffled[n_dev : n_dev + n_train]
    return train, dev


def save_clip_manifest(clips: list[ClipRef], path: Path) -> None:
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in clips], f, ensure_ascii=False, indent=2)

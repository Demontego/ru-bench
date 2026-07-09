import json
import random
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from ru_bench import config


@dataclass
class ClipRef:
    clip_id: str
    domain: str
    audio_path: str
    text: str
    duration: float


def download_golos_test_tar(dest: Path = config.GOLOS_TAR_PATH) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tar.part")
    urllib.request.urlretrieve(config.GOLOS_TEST_TAR_URL, tmp)
    tmp.rename(dest)
    return dest


def extract_golos_test_tar(tar_path: Path = config.GOLOS_TAR_PATH, dest_dir: Path = config.GOLOS_EXTRACT_DIR) -> Path:
    if dest_dir.exists():
        return dest_dir
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir.parent)
    return dest_dir


def load_domain_manifest(extract_dir: Path, domain: str) -> list[ClipRef]:
    manifest_path = extract_dir / domain / "manifest.jsonl"
    clips = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row["text"].strip():
                continue
            audio_path = str(extract_dir / domain / row["audio_filepath"])
            clips.append(
                ClipRef(
                    clip_id=row["id"],
                    domain=domain,
                    audio_path=audio_path,
                    text=row["text"],
                    duration=row["duration"],
                )
            )
    return clips


def load_all_manifests(extract_dir: Path = config.GOLOS_EXTRACT_DIR) -> list[ClipRef]:
    clips = []
    for domain in config.GOLOS_DOMAINS:
        clips.extend(load_domain_manifest(extract_dir, domain))
    return clips


def select_asr_sample(clips: list[ClipRef], n: int, seed: int = config.SEED) -> list[ClipRef]:
    rng = random.Random(seed)
    return rng.sample(clips, min(n, len(clips)))


def save_manifest(clips: list[ClipRef], path: Path = config.ASR_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in clips], f, ensure_ascii=False, indent=2)


def load_manifest(path: Path = config.ASR_MANIFEST_PATH) -> list[ClipRef]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [ClipRef(**row) for row in rows]

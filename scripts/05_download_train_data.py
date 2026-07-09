import argparse

from ru_bench import config
from ru_bench.train_data import (
    download_train_crowd_shard,
    download_train_farfield,
    load_clips_from_manifests,
    save_clip_manifest,
    select_train_dev_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["farfield", "crowd"], default="farfield")
    parser.add_argument("--crowd-shard", type=int, default=0, help="shard index 0-9, only used with --domain crowd")
    parser.add_argument("--n-train", type=int, default=config.TRAIN_SAMPLE_SIZE)
    parser.add_argument("--n-dev", type=int, default=config.DEV_SAMPLE_SIZE)
    args = parser.parse_args()

    if args.domain == "farfield":
        print("Downloading Golos train_farfield.tar (~15.4GB, cached if already present)...")
        extract_dir = download_train_farfield()
    else:
        print(f"Downloading Golos train_crowd{args.crowd_shard}.tar (cached if already present)...")
        extract_dir = download_train_crowd_shard(args.crowd_shard)

    clips = load_clips_from_manifests(extract_dir)
    print(f"Loaded {len(clips)} clips with matching audio")

    train, dev = select_train_dev_split(clips, args.n_train, args.n_dev)
    save_clip_manifest(train, config.TRAIN_MANIFEST_PATH)
    save_clip_manifest(dev, config.DEV_MANIFEST_PATH)
    print(f"Train: {len(train)} clips -> {config.TRAIN_MANIFEST_PATH}")
    print(f"Dev:   {len(dev)} clips -> {config.DEV_MANIFEST_PATH}")


if __name__ == "__main__":
    main()

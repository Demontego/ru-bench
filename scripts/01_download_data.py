import argparse

from ru_bench import config
from ru_bench.data import (
    download_golos_test_tar,
    extract_golos_test_tar,
    load_all_manifests,
    save_manifest,
    select_asr_sample,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-n", type=int, default=config.ASR_SAMPLE_SIZE)
    args = parser.parse_args()

    print("Downloading Golos test.tar (cached if already present)...")
    tar_path = download_golos_test_tar()
    print(f"Extracting {tar_path}...")
    extract_dir = extract_golos_test_tar(tar_path)

    clips = load_all_manifests(extract_dir)
    print(f"Loaded {len(clips)} clips from manifests ({config.GOLOS_DOMAINS})")

    sample = select_asr_sample(clips, args.asr_n)
    save_manifest(sample)
    print(f"Sampled {len(sample)} clips -> {config.ASR_MANIFEST_PATH}")


if __name__ == "__main__":
    main()

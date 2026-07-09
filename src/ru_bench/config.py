from pathlib import Path

SEED = 42

GOLOS_TEST_TAR_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos/test.tar"
GOLOS_DOMAINS = ["crowd", "farfield"]

ASR_SAMPLE_SIZE = 100

MAX_NEW_TOKENS_ASR = 4096

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

GOLOS_RAW_DIR = DATA_DIR / "raw" / "golos"
GOLOS_TAR_PATH = GOLOS_RAW_DIR / "test.tar"
GOLOS_EXTRACT_DIR = GOLOS_RAW_DIR / "test"

ASR_MANIFEST_PATH = DATA_DIR / "manifests" / "asr_sample.json"

ASR_RESULTS_DIR = RESULTS_DIR / "asr"
ASR_METRICS_PATH = RESULTS_DIR / "metrics_asr.json"
REPORT_PATH = RESULTS_DIR / "report.md"

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

# --- LoRA fine-tuning on Golos train split ---

GOLOS_TRAIN_FARFIELD_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_farfield.tar"
GOLOS_TRAIN_CROWD_SHARD_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos/train_crowd{shard}.tar"

GOLOS_TRAIN_RAW_DIR = DATA_DIR / "raw" / "golos_train"

TRAIN_SAMPLE_SIZE = 3000
DEV_SAMPLE_SIZE = 200

TRAIN_MANIFEST_PATH = DATA_DIR / "manifests" / "train_sample.json"
DEV_MANIFEST_PATH = DATA_DIR / "manifests" / "dev_sample.json"

# Whisper: full LoRA (attention + FFN). Qwen3: light LoRA (attention only).
LORA_R_WHISPER = 32
LORA_ALPHA_WHISPER = 64
LORA_R_QWEN = 8
LORA_ALPHA_QWEN = 16
LORA_DROPOUT = 0.05

CHECKPOINT_DIR = Path("checkpoints") / "lora_ru"

FINETUNED_ASR_RESULTS_DIR = RESULTS_DIR / "asr_finetuned"
FINETUNED_ASR_METRICS_PATH = RESULTS_DIR / "metrics_asr_finetuned.json"
FINETUNED_REPORT_PATH = RESULTS_DIR / "report_finetuned.md"

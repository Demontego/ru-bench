from pathlib import Path

SEED = 42

GOLOS_TEST_TAR_URL = "https://cdn.chatwm.opensmodel.sberdevices.ru/golos/test.tar"
GOLOS_DOMAINS = ["crowd", "farfield"]

ASR_SAMPLE_SIZE = 100

MAX_NEW_TOKENS_ASR = 4096

# ONNX/KV decode budgets (target text tokens, from train_sample+dev_sample).
# Diar ~30s clips: p99≈281 tok, max≈325; ~8–9.4 tok/s audio.
# Flat Golos ~10s: p99≈76. Headroom ×1.2 → round up.
MAX_NEW_TOKENS_CHUNK_10S = 128
MAX_NEW_TOKENS_CHUNK_30S = 384
TOKENS_PER_AUDIO_SEC_P99 = 9.5  # diar p99 ≈ 9.37

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
OPENSOURCE_RAW_DIR = DATA_DIR / "raw" / "opensource"

# 0 = use full pool (no artificial train/dev caps).
TRAIN_SAMPLE_SIZE = 0
DEV_SAMPLE_SIZE = 0
EVAL_RATIO = 0.10  # 90% train / 10% eval

# Domain -> language. Train is RU-heavy; eval prefers ~50/50 RU/EN for fair metrics.
EN_DOMAINS = frozenset({"fleurs_en", "librispeech_clean", "libri_convo_en"})
RU_DOMAINS = frozenset(
    {"farfield", "crowd", "golos10h", "fleurs_ru", "cv_ru", "synth_diar_ru"}
)

# Default mix: Golos + HF RU + EN flat retention + RU/EN multi-spk diar.
# Avoid empty HF repos (SberDevices/Golos); use bond005/sberdevices_golos_10h_crowd.
DEFAULT_TRAIN_SOURCES = [
    "golos_farfield",
    "golos10h",
    "fleurs_ru",
    "fleurs_en",
    "librispeech_clean",  # EN flat ASR retention
    "synth_diar_ru",  # RU multi-spk + timestamps
    "libri_convo_en",  # EN multi-spk + timestamps (format parity with RU diar)
]

# Per-source caps before 90/10 split.
# EN share kept high enough that forgetting is hard (not token-tiny).
# Flat EN trimmed slightly so diar EN can take real share without exploding pool.
SOURCE_LIMITS = {
    "golos_farfield": 4000,
    "golos10h": 2500,
    "fleurs_ru": 1500,
    "cv_ru": 2500,
    "fleurs_en": 2000,
    "librispeech_clean": 1500,
    "synth_diar_ru": 2000,
    "libri_convo_en": 1500,
}

TRAIN_MANIFEST_PATH = DATA_DIR / "manifests" / "train_sample.json"
DEV_MANIFEST_PATH = DATA_DIR / "manifests" / "dev_sample.json"

# RTX 3090 24GB LoRA recipe (Whisper LoRA r=32 + Qwen LoRA r=8 + vq_adaptor).
# Probe (longest ~13s Golos clips, bf16): bs=4 ~14GB OK; bs=8 peak~26GB >24GB — unsafe.
# With synth_diar_ru (~30s): bs=4 fills ~24GB and is slow — prefer bs=2.
# Safe default: microbatch=2, accum=8, eff=16. Override via CLI if needed.
TRAIN_BATCH_SIZE = 2
TRAIN_BATCH_ACCUM = 8  # effective batch = batch_size * batch_accum = 16
TRAIN_MAX_AUDIO_SECONDS = 30.0
TRAIN_VRAM_SAFE_MIB = 22000  # reject probe candidates above this peak

# Whisper: full LoRA (attention + FFN). Qwen3: light LoRA (attention only).
LORA_R_WHISPER = 32
LORA_ALPHA_WHISPER = 64
LORA_R_QWEN = 8
LORA_ALPHA_QWEN = 16
LORA_DROPOUT = 0.05

# Weighted CE for MOSS compact targets (speaker / timestamp / text only).
LOSS_W_SPEAKER = 4.0
LOSS_W_TIMESTAMP = 2.0
LOSS_W_TEXT = 1.0
LOSS_W_FLAT_WRAP = 0.4  # Golos wrap [0.00][S01]…[dur]
# Whole-clip multiplier for EN-domain examples (flat + diar).
LOSS_W_EN_CLIP = 1.5
# Fraction of microbatch slots drawn from EN (upsample vs natural ~10%).
TRAIN_EN_SAMPLE_RATIO = 0.30
# Within EN draws: fraction from multi-spk diar (libri_convo_en) vs flat ASR.
TRAIN_EN_DIAR_RATIO = 0.55

CHECKPOINT_DIR = Path("checkpoints") / "lora_ru"

FINETUNED_ASR_RESULTS_DIR = RESULTS_DIR / "asr_finetuned"
FINETUNED_ASR_METRICS_PATH = RESULTS_DIR / "metrics_asr_finetuned.json"
FINETUNED_REPORT_PATH = RESULTS_DIR / "report_finetuned.md"

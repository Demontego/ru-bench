"""Download / materialize open-source ASR clips into ClipRef manifests.

Sources (streaming HF where possible; audio written under data/raw/opensource/):
  - bond005/sberdevices_golos_10h_crowd  (RU; avoid empty SberDevices/Golos)
  - google/fleurs ru_ru / en_us
  - mozilla-foundation/common_voice_17_0 ru
  - openslr/librispeech_asr clean train.100 (optional EN)
  - ivkond/synthetic-speech-diarization-ru (RU multi-spk + timestamps)
  - gedeonmate/LibriConvo-segmented (EN multi-spk + timestamps, ≤30s)

Golos farfield/crowd CDN archives stay in train_data.py.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, DatasetDict, load_dataset
from datasets.data_files import EmptyDatasetError
from datasets.exceptions import DatasetNotFoundError

from ru_bench import config
from ru_bench.data import ClipRef

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


def _read_audio_field(audio_info: object) -> tuple[np.ndarray, int] | None:
    if not isinstance(audio_info, dict):
        return None
    if audio_info.get("array") is not None:
        return np.asarray(audio_info["array"], dtype=np.float32), int(audio_info["sampling_rate"])
    if audio_info.get("bytes") is not None:
        array, rate = sf.read(io.BytesIO(audio_info["bytes"]))
        return array.astype(np.float32), int(rate)
    path = audio_info.get("path")
    if path:
        array, rate = sf.read(path)
        return array.astype(np.float32), int(rate)
    return None


def _resample(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)


def _iter_hf_rows(
    path: str,
    *,
    config_name: str | None = None,
    split: str = "train",
    streaming: bool = True,
    decode_audio: bool | None = False,
) -> Iterator[dict]:
    """Yield HF rows. ``decode_audio=False`` tries bytes/path mode; ``None`` skips cast
    (needed when Hub audio is already arrays and torchcodec is missing)."""
    kwargs: dict = {"split": split, "streaming": streaming}
    if config_name is not None:
        kwargs["name"] = config_name
    ds = load_dataset(path, **kwargs)
    if isinstance(ds, DatasetDict):
        if split not in ds:
            raise ValueError(f"Split {split!r} missing in {path}: {list(ds.keys())}")
        ds = ds[split]
    if decode_audio is False:
        try:
            ds = ds.cast_column("audio", Audio(decode=False))
        except (AttributeError, TypeError, ValueError, ImportError):
            logger.debug("Could not disable audio decode for %s", path)
    yield from ds


def materialize_hf_source(
    source_id: str,
    *,
    hf_path: str,
    limit: int,
    out_dir: Path,
    domain: str,
    text_field: str,
    config_name: str | None = None,
    split: str = "train",
    row_filter: Callable[[dict], bool] | None = None,
    id_prefix: str | None = None,
) -> list[ClipRef]:
    """Stream HF rows, write wavs, return ClipRefs. Skips existing wavs (resume)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = id_prefix or source_id
    clips: list[ClipRef] = []
    seen = 0
    written = 0

    logger.info("Loading %s from %s (limit=%d)", source_id, hf_path, limit)
    try:
        rows = _iter_hf_rows(hf_path, config_name=config_name, split=split)
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Source %s unavailable: %s", source_id, exc)
        return []

    try:
        row_iter = iter(rows)
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Source %s unavailable at iterate: %s", source_id, exc)
        return []

    try:
        for row in row_iter:
            if written >= limit:
                break
            if row_filter is not None and not row_filter(row):
                continue

            text = str(row.get(text_field) or row.get("text") or row.get("sentence") or "").strip()
            if not text:
                continue

            raw_id = row.get("hash_id") or row.get("id") or row.get("path") or seen
            clip_id = f"{prefix}_{raw_id}".replace("/", "_").replace("\\", "_")
            wav_path = out_dir / f"{clip_id}.wav"
            seen += 1

            if wav_path.exists():
                try:
                    info = sf.info(str(wav_path))
                    duration = float(info.frames) / float(info.samplerate)
                except (OSError, RuntimeError):
                    continue
                clips.append(
                    ClipRef(
                        clip_id=clip_id,
                        domain=domain,
                        audio_path=str(wav_path),
                        text=text,
                        duration=duration,
                    )
                )
                written += 1
                continue

            decoded = _read_audio_field(row.get("audio"))
            if decoded is None:
                continue
            array, rate = decoded
            array = _resample(array, rate)
            if array.size < SAMPLE_RATE // 2:
                continue
            duration = float(array.size) / SAMPLE_RATE
            sf.write(str(wav_path), array, SAMPLE_RATE)
            clips.append(
                ClipRef(
                    clip_id=clip_id,
                    domain=domain,
                    audio_path=str(wav_path),
                    text=text,
                    duration=duration,
                )
            )
            written += 1
            if written % 200 == 0:
                logger.info("%s: materialized %d / %d", source_id, written, limit)
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Source %s aborted mid-stream: %s (kept %d)", source_id, exc, len(clips))

    logger.info("%s: done, %d clips", source_id, len(clips))
    return clips


def _safe_materialize(**kwargs) -> list[ClipRef]:
    """Wrap materialize so empty/gated HF repos don't abort the whole mix."""
    source_id = kwargs.get("source_id", "?")
    try:
        return materialize_hf_source(**kwargs)
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Source %s failed: %s — skipping", source_id, exc)
        return []


def load_golos10h(limit: int, raw_dir: Path = config.OPENSOURCE_RAW_DIR) -> list[ClipRef]:
    return _safe_materialize(
        source_id="golos10h",
        hf_path="bond005/sberdevices_golos_10h_crowd",
        limit=limit,
        out_dir=raw_dir / "golos10h",
        domain="golos10h",
        text_field="transcription",
    )


def load_fleurs_ru(limit: int, raw_dir: Path = config.OPENSOURCE_RAW_DIR) -> list[ClipRef]:
    return _safe_materialize(
        source_id="fleurs_ru",
        hf_path="google/fleurs",
        config_name="ru_ru",
        limit=limit,
        out_dir=raw_dir / "fleurs_ru",
        domain="fleurs_ru",
        text_field="transcription",
    )


def load_fleurs_en(limit: int, raw_dir: Path = config.OPENSOURCE_RAW_DIR) -> list[ClipRef]:
    return _safe_materialize(
        source_id="fleurs_en",
        hf_path="google/fleurs",
        config_name="en_us",
        limit=limit,
        out_dir=raw_dir / "fleurs_en",
        domain="fleurs_en",
        text_field="transcription",
    )


def load_common_voice_ru(limit: int, raw_dir: Path = config.OPENSOURCE_RAW_DIR) -> list[ClipRef]:
    # mozilla-foundation/common_voice_17_0 often empty/gated on Hub (EmptyDatasetError).
    # Soft-skip; mix still works with golos10h + FLEURS.
    return _safe_materialize(
        source_id="cv_ru",
        hf_path="mozilla-foundation/common_voice_17_0",
        config_name="ru",
        limit=limit,
        out_dir=raw_dir / "cv_ru",
        domain="cv_ru",
        text_field="sentence",
        row_filter=lambda row: int(row.get("down_votes", 0) or 0)
        <= int(row.get("up_votes", 0) or 0),
    )


def load_librispeech_clean(limit: int, raw_dir: Path = config.OPENSOURCE_RAW_DIR) -> list[ClipRef]:
    return _safe_materialize(
        source_id="librispeech_clean",
        hf_path="openslr/librispeech_asr",
        config_name="clean",
        split="train.100",
        limit=limit,
        out_dir=raw_dir / "librispeech_clean",
        domain="librispeech_clean",
        text_field="text",
    )


def _truncate_audio(array: np.ndarray, max_seconds: float) -> np.ndarray:
    max_samples = int(max_seconds * SAMPLE_RATE)
    if array.size <= max_samples:
        return array
    return array[:max_samples]


def load_synth_diar_ru(
    limit: int,
    raw_dir: Path = config.OPENSOURCE_RAW_DIR,
    *,
    max_audio_seconds: float = config.TRAIN_MAX_AUDIO_SECONDS,
    min_speakers: int = 2,
    min_segments: int = 2,
) -> list[ClipRef]:
    """Materialize ivkond/synthetic-speech-diarization-ru with MOSS multi-segment targets.

    Format retention: teaches ``[t0][Sxx]text[t1]`` turns instead of flat Golos wrap.
    Long tracks are truncated to ``max_audio_seconds`` (VRAM); segments clipped to match.
    """
    from ru_bench.moss_format import speakers_to_moss_target

    out_dir = raw_dir / "synth_diar_ru"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ClipRef] = []
    written = 0
    seen = 0

    logger.info(
        "Loading synth_diar_ru (limit=%d, max_s=%.1f, min_spk=%d)",
        limit,
        max_audio_seconds,
        min_speakers,
    )
    try:
        rows = _iter_hf_rows(
            "ivkond/synthetic-speech-diarization-ru",
            split="train",
            decode_audio=None,  # parquet embeds arrays; cast(decode=False) needs torchcodec
        )
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("synth_diar_ru unavailable: %s", exc)
        return []

    try:
        for row in rows:
            if written >= limit:
                break
            seen += 1
            num_spk = int(row.get("num_speakers") or 0)
            speakers = row.get("speakers") or []
            if num_spk < min_speakers or len(speakers) < min_segments:
                continue

            target = speakers_to_moss_target(speakers, max_end=max_audio_seconds)
            if target.count("[S") < min_segments:
                continue

            merged_text = " ".join(
                str(seg.get("text") or "").strip()
                for seg in sorted(speakers, key=lambda x: float(x["start"]))
                if float(seg["start"]) < max_audio_seconds
                and str(seg.get("text") or "").strip()
            )
            if not merged_text:
                continue

            clip_id = f"synth_diar_ru_{seen:05d}"
            wav_path = out_dir / f"{clip_id}.wav"

            if wav_path.exists():
                try:
                    info = sf.info(str(wav_path))
                    duration = float(info.frames) / float(info.samplerate)
                except (OSError, RuntimeError):
                    continue
                clips.append(
                    ClipRef(
                        clip_id=clip_id,
                        domain="synth_diar_ru",
                        audio_path=str(wav_path),
                        text=merged_text,
                        duration=duration,
                        target=target,
                    )
                )
                written += 1
                continue

            decoded = _read_audio_field(row.get("audio"))
            if decoded is None:
                continue
            array, rate = decoded
            array = _resample(array, rate)
            array = _truncate_audio(array, max_audio_seconds)
            if array.size < SAMPLE_RATE // 2:
                continue
            duration = float(array.size) / SAMPLE_RATE
            sf.write(str(wav_path), array, SAMPLE_RATE)
            clips.append(
                ClipRef(
                    clip_id=clip_id,
                    domain="synth_diar_ru",
                    audio_path=str(wav_path),
                    text=merged_text,
                    duration=duration,
                    target=target,
                )
            )
            written += 1
            if written % 100 == 0:
                logger.info("synth_diar_ru: materialized %d / %d", written, limit)
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("synth_diar_ru aborted mid-stream: %s (kept %d)", exc, len(clips))

    logger.info("synth_diar_ru: done, %d clips", len(clips))
    return clips


def load_libri_convo_en(
    limit: int,
    raw_dir: Path = config.OPENSOURCE_RAW_DIR,
    *,
    max_audio_seconds: float = config.TRAIN_MAX_AUDIO_SECONDS,
    min_speakers: int = 2,
    min_segments: int = 2,
    split: str = "train",
) -> list[ClipRef]:
    """Materialize LibriConvo-segmented EN dialogues with MOSS multi-segment targets.

    Each row is a ≤30s conversation fragment with parallel start/end/text/speaker lists.
    """
    from ru_bench.moss_format import format_moss_segment

    out_dir = raw_dir / "libri_convo_en"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips: list[ClipRef] = []
    written = 0
    seen = 0

    logger.info(
        "Loading libri_convo_en (limit=%d, max_s=%.1f, min_spk=%d)",
        limit,
        max_audio_seconds,
        min_speakers,
    )
    try:
        rows = _iter_hf_rows(
            "gedeonmate/LibriConvo-segmented",
            split=split,
            decode_audio=False,
        )
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("libri_convo_en unavailable: %s", exc)
        return []

    try:
        for row in rows:
            if written >= limit:
                break
            seen += 1
            starts = list(row.get("start_time") or [])
            ends = list(row.get("end_time") or [])
            texts = list(row.get("text") or [])
            symbols = list(row.get("abstract_symbol") or [])
            n = min(len(starts), len(ends), len(texts), len(symbols))
            if n < min_segments:
                continue

            spk_map: dict[str, int] = {}
            parts: list[str] = []
            flat: list[str] = []
            for i in range(n):
                start = float(starts[i])
                end = float(ends[i])
                if start >= max_audio_seconds:
                    break
                end = min(end, max_audio_seconds)
                text = str(texts[i] or "").strip()
                if not text or end <= start:
                    continue
                sym = str(symbols[i] or "").strip() or "?"
                if sym not in spk_map:
                    spk_map[sym] = len(spk_map) + 1
                sid = spk_map[sym]
                parts.append(format_moss_segment(start, end, f"S{sid:02d}", text))
                flat.append(text)
            if len(spk_map) < min_speakers or len(parts) < min_segments:
                continue
            target = "".join(parts)
            merged_text = " ".join(flat)
            if not merged_text:
                continue

            seg_id = row.get("segment_conversation_id") or row.get("conversation_id") or seen
            clip_id = f"libri_convo_en_{str(seg_id).replace('/', '_')}"
            wav_path = out_dir / f"{clip_id}.wav"

            if wav_path.exists():
                try:
                    info = sf.info(str(wav_path))
                    duration = float(info.frames) / float(info.samplerate)
                except (OSError, RuntimeError):
                    continue
                clips.append(
                    ClipRef(
                        clip_id=clip_id,
                        domain="libri_convo_en",
                        audio_path=str(wav_path),
                        text=merged_text,
                        duration=duration,
                        target=target,
                    )
                )
                written += 1
                continue

            decoded = _read_audio_field(row.get("audio"))
            if decoded is None:
                continue
            array, rate = decoded
            array = _resample(array, rate)
            array = _truncate_audio(array, max_audio_seconds)
            if array.size < SAMPLE_RATE // 2:
                continue
            duration = float(array.size) / SAMPLE_RATE
            sf.write(str(wav_path), array, SAMPLE_RATE)
            clips.append(
                ClipRef(
                    clip_id=clip_id,
                    domain="libri_convo_en",
                    audio_path=str(wav_path),
                    text=merged_text,
                    duration=duration,
                    target=target,
                )
            )
            written += 1
            if written % 20 == 0:
                logger.info("libri_convo_en: materialized %d / %d", written, limit)
    except (DatasetNotFoundError, EmptyDatasetError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("libri_convo_en aborted mid-stream: %s (kept %d)", exc, len(clips))

    logger.info("libri_convo_en: done, %d clips", len(clips))
    return clips


SOURCE_LOADERS: dict[str, Callable[[int], list[ClipRef]]] = {
    "golos10h": load_golos10h,
    "fleurs_ru": load_fleurs_ru,
    "fleurs_en": load_fleurs_en,
    "cv_ru": load_common_voice_ru,
    "librispeech_clean": load_librispeech_clean,
    "synth_diar_ru": load_synth_diar_ru,
    "libri_convo_en": load_libri_convo_en,
}


def load_opensource_sources(
    sources: list[str],
    limits: dict[str, int] | None = None,
) -> list[ClipRef]:
    limits = limits or config.SOURCE_LIMITS
    clips: list[ClipRef] = []
    for source in sources:
        if source not in SOURCE_LOADERS:
            continue
        limit = int(limits.get(source, 1000))
        part = SOURCE_LOADERS[source](limit)
        clips.extend(part)
        print(f"  {source}: {len(part)} clips")
    return clips

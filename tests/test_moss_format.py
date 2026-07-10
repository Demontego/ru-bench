"""Tests for MOSS compact transcript formatting."""

from ru_bench.data import ClipRef
from ru_bench.moss_format import speakers_to_moss_target, target_text


def test_speakers_to_moss_target_multi() -> None:
    speakers = [
        {"speaker_id": 1, "start": 0.0, "end": 1.5, "text": "привет"},
        {"speaker_id": 2, "start": 1.5, "end": 3.0, "text": "здравствуйте"},
    ]
    got = speakers_to_moss_target(speakers)
    assert got == "[0.00][S01]привет[1.50][1.50][S02]здравствуйте[3.00]"


def test_speakers_to_moss_target_truncates() -> None:
    speakers = [
        {"speaker_id": 1, "start": 0.0, "end": 2.0, "text": "a"},
        {"speaker_id": 2, "start": 25.0, "end": 40.0, "text": "b long"},
        {"speaker_id": 1, "start": 41.0, "end": 45.0, "text": "c"},
    ]
    got = speakers_to_moss_target(speakers, max_end=30.0)
    assert got == "[0.00][S01]a[2.00][25.00][S02]b long[30.00]"
    assert "S01]c" not in got


def test_target_text_prefers_stored() -> None:
    clip = ClipRef(
        clip_id="x",
        domain="synth_diar_ru",
        audio_path="x.wav",
        text="merged",
        duration=3.0,
        target="[0.00][S01]hi[1.00][1.00][S02]bye[3.00]",
    )
    assert target_text(clip).startswith("[0.00][S01]hi")


def test_target_text_flat_fallback() -> None:
    clip = ClipRef(
        clip_id="y",
        domain="farfield",
        audio_path="y.wav",
        text="один",
        duration=1.25,
    )
    assert target_text(clip) == "[0.00][S01]один[1.25]"

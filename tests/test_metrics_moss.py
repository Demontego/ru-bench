"""Tests for weighted CE (spk/ts/text) + MOSS metrics."""

from moss_transcribe_diarize.transcript_parser import TranscriptSegment

from ru_bench.data import ClipRef
from ru_bench.loss_weights import LossWeightConfig, _bracket_spans, target_token_weights
from ru_bench.metrics_moss import cer_speaker_independent, cp_cer, delta_cp, score_transcript_pair
from ru_bench.moss_format import speakers_to_moss_target, target_text


def test_cer_cpcer_perfect() -> None:
    ref = [
        TranscriptSegment(0.0, 1.0, "S01", "привет"),
        TranscriptSegment(1.0, 2.0, "S02", "мир"),
    ]
    s = score_transcript_pair(ref, ref)
    assert s["cer"] == 0.0 and s["cp_cer"] == 0.0 and s["delta_cp"] == 0.0


def test_cpcer_speaker_swap() -> None:
    ref = [
        TranscriptSegment(0.0, 1.0, "S01", "аааа"),
        TranscriptSegment(1.0, 2.0, "S02", "бббб"),
    ]
    hyp = [
        TranscriptSegment(0.0, 1.0, "S02", "аааа"),
        TranscriptSegment(1.0, 2.0, "S01", "бббб"),
    ]
    assert cer_speaker_independent(ref, hyp) == 0.0
    assert cp_cer(ref, hyp) == 0.0
    assert delta_cp(0.0, 0.0) == 0.0


def test_bracket_spans_kinds() -> None:
    spans = _bracket_spans("[0.00][S01]hi[1.50]")
    kinds = [k for _, _, k in spans]
    assert kinds == ["timestamp", "speaker", "timestamp"]


def test_spk_ts_text_weights() -> None:
    class FakeTok:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            ids = list(range(len(text)))
            offsets = [(i, i + 1) for i in range(len(text))]
            return {"input_ids": ids, "offset_mapping": offsets}

    cfg = LossWeightConfig(speaker=4.0, timestamp=2.0, text=1.0, flat_wrap=0.4)
    target = "[1.00][S01]hi![2.00]"
    w = target_token_weights(FakeTok(), target, cfg=cfg, is_flat=False)
    assert w[target.index("1")] == 2.0  # timestamp digit
    assert w[target.index("S")] == 4.0
    assert w[target.index("h")] == 1.0
    assert w[target.index("!")] == 1.0


def test_flat_wrap_downweight() -> None:
    class FakeTok:
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            ids = list(range(len(text)))
            offsets = [(i, i + 1) for i in range(len(text))]
            return {"input_ids": ids, "offset_mapping": offsets}

    cfg = LossWeightConfig(speaker=4.0, timestamp=2.0, text=1.0, flat_wrap=0.4)
    target = "[0.00][S01]один[1.25]"
    w = target_token_weights(FakeTok(), target, cfg=cfg, is_flat=True)
    assert w[target.index("0")] == 0.4
    assert w[target.index("S")] == 0.4
    assert w[target.index("о")] == 1.0


def test_speakers_to_moss_target_multi() -> None:
    speakers = [
        {"speaker_id": 1, "start": 0.0, "end": 1.5, "text": "привет"},
        {"speaker_id": 2, "start": 1.5, "end": 3.0, "text": "здравствуйте"},
    ]
    got = speakers_to_moss_target(speakers)
    assert got == "[0.00][S01]привет[1.50][1.50][S02]здравствуйте[3.00]"


def test_target_text_flat_fallback() -> None:
    clip = ClipRef("y", "farfield", "y.wav", "один", 1.25)
    assert target_text(clip) == "[0.00][S01]один[1.25]"

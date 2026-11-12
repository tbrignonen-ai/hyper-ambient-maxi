"""Endpointing and push-to-talk, using synthetic PCM. No microphone."""

import math
from array import array

from hyper_ambient_maxi.ears.ptt import PushToTalk
from hyper_ambient_maxi.ears.vad import speech_spans, trim_speech
from hyper_ambient_maxi.ears.wavutil import SAMPLE_RATE

FRAME = 320  # 20 ms at 16 kHz


def _pcm(samples: list[int]) -> bytes:
    buf = array("h", samples)
    return buf.tobytes()


def _tone(frames: int, amplitude: int = 8000) -> list[int]:
    count = frames * FRAME
    return [
        int(amplitude * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE))
        for index in range(count)
    ]


def _quiet(frames: int) -> list[int]:
    return [0] * (frames * FRAME)


def test_a_tone_between_silence_is_one_span():
    samples = _quiet(100) + _tone(20) + _quiet(30)  # 2.0s + 0.4s + 0.6s
    spans = speech_spans(_pcm(samples))
    assert len(spans) == 1
    start, end = spans[0]
    assert abs(start - 100 * FRAME) < FRAME
    assert abs(end - 120 * FRAME) < 2 * FRAME
    trimmed = trim_speech(_pcm(samples))
    assert trimmed is not None
    assert len(trimmed) < len(_pcm(samples))


def test_two_bursts_stay_two_spans_and_a_click_is_dropped():
    samples = _quiet(10) + _tone(20) + _quiet(30) + _tone(15) + _quiet(25)
    assert len(speech_spans(_pcm(samples))) == 2
    click = _quiet(10) + _tone(3) + _quiet(30)  # 60 ms, under the 100 ms start
    assert speech_spans(_pcm(click)) == []
    assert trim_speech(_pcm(click)) is None


def test_push_to_talk_ignores_audio_until_the_key_is_held():
    talk = PushToTalk()
    talk.feed(_pcm(_tone(20)))
    assert talk.release() is None
    talk.press()
    talk.feed(_pcm(_quiet(5) + _tone(20) + _quiet(25)))
    spoken = talk.release()
    assert spoken is not None
    assert len(spoken) < len(_pcm(_quiet(5) + _tone(20) + _quiet(25)))
    talk.press()
    talk.feed(_pcm(_tone(2)))
    assert talk.release() is None

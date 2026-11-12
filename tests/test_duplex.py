"""Barge-in on fixture PCM. No sound device."""

import math
from array import array

from hyper_ambient_maxi.ears.duplex import FRAME_SAMPLES, simulate
from hyper_ambient_maxi.ears.wavutil import SAMPLE_RATE


def _pcm(samples: list[int]) -> bytes:
    return array("h", samples).tobytes()


def _tone(frames: int, amplitude: int = 8000) -> bytes:
    count = frames * FRAME_SAMPLES
    return _pcm(
        [
            int(amplitude * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE))
            for index in range(count)
        ]
    )


def _quiet(frames: int) -> bytes:
    return _pcm([0] * (frames * FRAME_SAMPLES))


def test_silence_lets_playback_finish():
    playback = _tone(50)
    result = simulate(playback, _quiet(50))
    assert result.interrupted is False
    assert result.capture is None
    assert result.played_frames == result.total_frames
    assert result.played == playback


def test_speech_cancels_the_rest_of_playback():
    playback = _tone(50)
    microphone = _quiet(10) + _tone(20) + _quiet(20)
    result = simulate(playback, microphone)
    assert result.interrupted is True
    assert result.played_frames < result.total_frames
    assert result.capture is not None
    assert len(result.played) < len(playback)


def test_a_click_does_not_barge_in():
    playback = _tone(40)
    microphone = _quiet(10) + _tone(3) + _quiet(30)
    result = simulate(playback, microphone)
    assert result.interrupted is False
    assert result.played_frames == result.total_frames

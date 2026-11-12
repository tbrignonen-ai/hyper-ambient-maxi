"""Energy endpointing for a whole utterance.

This is the L1 turn detector Hyper Ambient keeps as a fallback when no
learned model is loaded: speech starts after a short run of loud frames
and ends after a run of quiet ones. Frames are 20 ms at 16 kHz. No
neural net, so the tests can build the WAV themselves.
"""

from __future__ import annotations

from hyper_ambient_maxi.ears.wavutil import SILENCE_RMS, rms

FRAME_MS = 20
SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
# A click shorter than this is not a turn.
MIN_SPEECH_MS = 100
# Quiet this long after speech closes the turn.
END_SILENCE_MS = 400


def speech_spans(pcm: bytes, rate: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """Return half-open sample ranges that contain speech."""
    if rate != SAMPLE_RATE:
        raise ValueError("endpointing expects 16 kHz mono pcm")
    frame_bytes = FRAME_SAMPLES * 2
    if frame_bytes <= 0 or len(pcm) < frame_bytes:
        return []
    loud_needed = max(1, MIN_SPEECH_MS // FRAME_MS)
    quiet_needed = max(1, END_SILENCE_MS // FRAME_MS)
    spans: list[tuple[int, int]] = []
    in_speech = False
    loud_run = 0
    quiet_run = 0
    start = 0
    frames = len(pcm) // frame_bytes
    for index in range(frames):
        chunk = pcm[index * frame_bytes : (index + 1) * frame_bytes]
        loud = rms(chunk) >= SILENCE_RMS
        if not in_speech:
            loud_run = loud_run + 1 if loud else 0
            if loud_run >= loud_needed:
                in_speech = True
                start = (index - loud_run + 1) * FRAME_SAMPLES
                quiet_run = 0
            continue
        if loud:
            quiet_run = 0
            continue
        quiet_run += 1
        if quiet_run >= quiet_needed:
            end = (index - quiet_run + 1) * FRAME_SAMPLES
            if end > start:
                spans.append((start, end))
            in_speech = False
            loud_run = 0
            quiet_run = 0
    if in_speech:
        spans.append((start, frames * FRAME_SAMPLES))
    return spans


def trim_speech(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes | None:
    spans = speech_spans(pcm, rate)
    if not spans:
        return None
    start = spans[0][0] * 2
    end = spans[-1][1] * 2
    return pcm[start:end]

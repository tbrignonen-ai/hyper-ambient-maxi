"""WAV framing, energy, and a junk-transcript filter.

Adapted from hyper-ambient-mini (Apache-2.0), ``app/mini_harness.py``:
``as_samples``, ``rms_db_of``, ``resample_to_16k``, ``wav_bytes``, and
``is_junk``. Original: https://github.com/tbrignonen-ai/hyper-ambient-mini
The mini harness is push-to-talk dictation into the clipboard. These
helpers are the part that does not touch a microphone, a window, or a
Windows parakeet.exe. Copyright of the original helpers belongs to
their contributors. The adaptations in this file are Apache-2.0.
"""

from __future__ import annotations

import array
import io
import wave

SAMPLE_RATE = 16000
# int16 RMS. Room tone sits well under this; a voiced frame does not.
# The mini app uses the RMS as a meter, not as a gate. The gate is ours,
# so silence cannot become a memory.
SILENCE_RMS = 150.0

# Whisper-style recited credits. Dropped in mini before they hit the
# clipboard; dropped here before they hit the memory journal.
JUNK_PATTERNS = (
    "subtitles by",
    "sous titrage",
    "amara org",
    "thanks for watching",
    "abonnez vous",
    "merci d avoir regarde",
    "subtitle",
    "www",
)


def as_samples(raw: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(bytes(raw[: len(raw) - (len(raw) % 2)]))
    return samples


def rms(raw: bytes) -> float:
    """RMS of an int16 frame. The array module keeps numpy optional."""
    samples = as_samples(raw)
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return (total / len(samples)) ** 0.5


def resample_to_16k(raw: bytes, src_rate: int) -> bytes:
    """Linear resample of int16 mono to 16 kHz."""
    if src_rate == SAMPLE_RATE:
        return bytes(raw)
    incoming = as_samples(raw)
    if not incoming:
        return b""
    step = src_rate / SAMPLE_RATE
    count = int(len(incoming) / step)
    outgoing = array.array("h", bytes(2 * count))
    last = len(incoming) - 1
    for index in range(count):
        pos = index * step
        left = int(pos)
        frac = pos - left
        right = left + 1 if left < last else left
        sample = (1 - frac) * incoming[left] + frac * incoming[right]
        outgoing[index] = int(sample)
    return outgoing.tobytes()


def wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def pcm_from_wav(blob: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(blob), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError("Maxi expects 16-bit mono WAV")
        rate = handle.getframerate()
        pcm = handle.readframes(handle.getnframes())
    if rate != SAMPLE_RATE:
        pcm = resample_to_16k(pcm, rate)
        rate = SAMPLE_RATE
    return pcm, rate


def is_junk(text: str) -> bool:
    folded = " ".join(text.lower().split())
    if len(folded) < 2:
        return True
    if any(pattern in folded for pattern in JUNK_PATTERNS):
        return True
    words = [word.strip(".,!?") for word in folded.split() if word.strip(".,!?")]
    for size in (1, 2, 3):
        if len(words) < size * 3:
            continue
        groups = [
            tuple(words[index : index + size])
            for index in range(0, len(words) - size + 1, size)
        ]
        if len(groups) >= 3 and len(set(groups)) == 1:
            return True
    return False

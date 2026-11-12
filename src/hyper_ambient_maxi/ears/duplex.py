"""Streaming playback that the user can interrupt.

One step is one 20 ms frame. While the mouth is speaking, a run of loud
microphone frames cancels the rest of the buffer. The caller supplies
both streams. Nothing here opens a sound device, so the tests are WAV
fixtures and generated PCM.
"""

from __future__ import annotations

from dataclasses import dataclass

from hyper_ambient_maxi.ears.vad import FRAME_MS, FRAME_SAMPLES, MIN_SPEECH_MS
from hyper_ambient_maxi.ears.wavutil import SILENCE_RMS, rms

FRAME_BYTES = FRAME_SAMPLES * 2
LOUD_FRAMES = max(1, MIN_SPEECH_MS // FRAME_MS)


@dataclass
class DuplexResult:
    played: bytes
    interrupted: bool
    capture: bytes | None
    played_frames: int
    total_frames: int


class Duplex:
    def __init__(self) -> None:
        self.phase = "idle"
        self.interrupted = False
        self._play: list[bytes] = []
        self._cursor = 0
        self._loud = 0
        self._capture = bytearray()

    @property
    def total_frames(self) -> int:
        return len(self._play)

    @property
    def played_frames(self) -> int:
        return self._cursor

    def start_speaking(self, pcm: bytes) -> None:
        self.phase = "speaking"
        self.interrupted = False
        self._play = split_frames(pcm)
        self._cursor = 0
        self._loud = 0
        self._capture.clear()

    def interrupt(self) -> None:
        if self.phase == "speaking":
            self.interrupted = True
            self.phase = "idle"

    def step(self, mic_frame: bytes) -> bytes | None:
        """Take one mic frame. Return the next playback frame, if any."""
        frame = _one_frame(mic_frame)
        if self.phase != "speaking":
            return None
        if rms(frame) >= SILENCE_RMS:
            self._loud += 1
            self._capture.extend(frame)
        else:
            self._loud = 0
            self._capture.clear()
        if self._loud >= LOUD_FRAMES:
            self.interrupted = True
            self.phase = "idle"
            return None
        if self._cursor >= len(self._play):
            self.phase = "idle"
            return None
        out = self._play[self._cursor]
        self._cursor += 1
        return out

    def capture(self) -> bytes | None:
        if not self.interrupted or not self._capture:
            return None
        return bytes(self._capture)


def simulate(playback: bytes, microphone: bytes) -> DuplexResult:
    """Align mic frames with playback frames and run until barge-in or the end."""
    session = Duplex()
    session.start_speaking(playback)
    mic_frames = split_frames(microphone) or [silence_frame()]
    played = bytearray()
    index = 0
    while session.phase == "speaking":
        frame = mic_frames[index] if index < len(mic_frames) else silence_frame()
        index += 1
        out = session.step(frame)
        if out:
            played.extend(out)
        if index > session.total_frames + len(mic_frames) + 2:
            break
    return DuplexResult(
        played=bytes(played),
        interrupted=session.interrupted,
        capture=session.capture(),
        played_frames=session.played_frames,
        total_frames=session.total_frames,
    )


def split_frames(pcm: bytes) -> list[bytes]:
    if not pcm:
        return []
    frames = []
    for start in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        frames.append(pcm[start : start + FRAME_BYTES])
    return frames


def silence_frame() -> bytes:
    return b"\x00\x00" * FRAME_SAMPLES


def loud_frame() -> bytes:
    return (8000).to_bytes(2, "little", signed=True) * FRAME_SAMPLES


def _one_frame(pcm: bytes) -> bytes:
    if len(pcm) >= FRAME_BYTES:
        return pcm[:FRAME_BYTES]
    return pcm + (b"\x00\x00" * ((FRAME_BYTES - len(pcm)) // 2))

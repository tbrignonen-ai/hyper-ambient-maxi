"""Push-to-talk as a state machine, with no keyboard and no microphone.

Hyper Ambient Mini holds a key, buffers PCM, and flushes on release.
The same contract is tested here by calling press, feed, and release.
Audio that arrives while the key is up is ignored. A release with no
speech span returns nothing, so a click cannot become a memory.
"""

from __future__ import annotations

from hyper_ambient_maxi.ears.vad import trim_speech


class PushToTalk:
    def __init__(self) -> None:
        self.held = False
        self._buffer = bytearray()

    def press(self) -> None:
        self.held = True
        self._buffer.clear()

    def feed(self, pcm: bytes) -> None:
        if self.held and pcm:
            self._buffer.extend(pcm)

    def release(self) -> bytes | None:
        self.held = False
        pcm = bytes(self._buffer)
        self._buffer.clear()
        if not pcm:
            return None
        return trim_speech(pcm)

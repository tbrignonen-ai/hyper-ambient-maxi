"""A presence loop with no window and no microphone.

States are idle, listening, thinking, and speaking. ``/hold`` is the
push-to-talk press. ``/release <words>`` is the utterance, because the
tests do not run speech recognition. ``/say <words>`` is a text turn.
Holding the key while the mouth is speaking cancels the rest of the
playback buffer. That is the barge-in the duplex clock already tests
on PCM; here the same ``Duplex`` object is what a hold interrupts.
"""

from __future__ import annotations

import math
from array import array

from hyper_ambient_maxi.ears.duplex import LOUD_FRAMES, Duplex, loud_frame
from hyper_ambient_maxi.ears.hotkey import ScriptedHotkey
from hyper_ambient_maxi.ears.ptt import PushToTalk
from hyper_ambient_maxi.ears.wavutil import SAMPLE_RATE  # reply tone length
from hyper_ambient_maxi.loop import run_turn


class PresenceSession:
    def __init__(self, store, brain) -> None:
        self.store = store
        self.brain = brain
        self.state = "idle"
        self.ptt = PushToTalk()
        self.duplex = Duplex()
        self.last_reply = ""
        self.last_memory = ""
        self.cuts = 0
        self.last_cut: tuple[int, int, bool] | None = None

    def command(self, line: str) -> list[str]:
        text = " ".join((line or "").split())
        if not text:
            return []
        if text == "/hold" or text.startswith("/hold "):
            return self._hold()
        if text.startswith("/release"):
            return self._release(text[len("/release") :].strip())
        if text.startswith("/say "):
            return self._say(text[len("/say ") :].strip())
        return self._say(text)

    def run_script(self, script: str) -> list[str]:
        out: list[str] = []
        for line in script.splitlines():
            out.extend(self.command(line))
        return out

    def drive_hotkey(self, hotkey: ScriptedHotkey, on_release: dict[int, str]) -> list[str]:
        """Pair each release with the next injected transcript."""
        out: list[str] = []
        release_index = 0
        while True:
            event = hotkey.poll()
            if event is None:
                break
            if event == "press":
                out.extend(self._hold())
            elif event == "release":
                out.extend(self._release(on_release.get(release_index, "")))
                release_index += 1
        return out

    def _hold(self) -> list[str]:
        lines: list[str] = []
        lines.extend(self._cancel_if_speaking())
        self.ptt.press()
        self.state = "listening"
        lines.append("[listening]")
        return lines

    def _release(self, utterance: str) -> list[str]:
        self.ptt.release()
        if not utterance:
            self.state = "idle"
            return ["[idle]"]
        return self._turn(utterance)

    def _say(self, utterance: str) -> list[str]:
        lines: list[str] = []
        lines.extend(self._cancel_if_speaking())
        if not utterance:
            self.state = "idle"
            lines.append("[idle]")
            return lines
        lines.extend(self._turn(utterance))
        return lines

    def _cancel_if_speaking(self) -> list[str]:
        if self.state != "speaking":
            return []
        for _ in range(LOUD_FRAMES):
            self.duplex.step(loud_frame())
        self.cuts += 1
        self.last_cut = (
            self.duplex.played_frames,
            self.duplex.total_frames,
            self.duplex.interrupted,
        )
        return ["[barge-in]"]

    def _turn(self, utterance: str) -> list[str]:
        self.state = "thinking"
        result = run_turn(
            self.store,
            self.brain,
            utterance,
            conversation_id="presence",
            source_agent="maxi-presence",
        )
        self.last_reply = result.reply
        self._note_memory(result)
        self.duplex.start_speaking(_reply_tone(result.reply))
        self.state = "speaking"
        flag = "saved" if result.memory_ok else "NOT SAVED"
        return [f"[thinking]", f"[speaking] {result.reply}", f"[memory] {flag}"]

    def snapshot(self) -> str:
        """What the window shows. No Tk, no display."""
        return "\n".join(
            [
                f"state: {self.state}",
                f"reply: {self.last_reply or '(none)'}",
                f"memory: {self.last_memory or '(none)'}",
            ]
        )

    def _note_memory(self, result) -> None:
        receipt = result.receipt
        if receipt is None:
            return
        if receipt.facts:
            self.last_memory = receipt.facts[-1].summary
        elif receipt.already:
            self.last_memory = receipt.already[0].summary


def _reply_tone(reply: str) -> bytes:
    """A second of tone stands in for TTS. The duplex clock can cancel it."""
    del reply
    samples = array("h")
    count = SAMPLE_RATE
    for index in range(count):
        samples.append(int(4000 * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE)))
    return samples.tobytes()

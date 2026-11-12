"""Piper speaks, faster-whisper hears, the scripted brain stores it.

Skipped when the packages or the cached voices are absent, so a machine
without those files keeps the core suite light. No microphone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.ears.local_speech import (
    faster_whisper_ready,
    piper_ready,
    recognize,
    synthesize,
)
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.store import MemoryStore
from hyper_ambient_maxi.voice import run_audio_turn

pytestmark = pytest.mark.skipif(
    not (faster_whisper_ready() and piper_ready()),
    reason="faster-whisper or Piper voices are not in the local cache",
)

ROOT = Path(__file__).resolve().parents[1]
EN = "My dog is Pixel."
FR = "Je l'ai renommé Biscuit."


def test_voice_loop_round_trip_and_latency(tmp_path):
    en_wav, en_tts = _spoken(EN)
    fr_wav, fr_tts = _spoken(FR)
    assert en_tts < 5
    assert fr_tts < 5

    store = MemoryStore(tmp_path / "home")
    brain = ScriptedBrain()
    heard_en, en_stt = _heard(en_wav, language="en")
    saved = run_audio_turn(
        store,
        brain,
        en_wav,
        ears="faster-whisper",
        mouth="piper",
        language="en",
        initial_prompt="My dog is Pixel. Biscuit.",
    )
    assert saved.status == "ok", saved.reason
    assert saved.turn is not None and saved.turn.memory_ok
    dog = store.slot("fact:dog-name")
    assert dog is not None and _close(dog.value, "Pixel"), dog.value if dog else None
    assert saved.audio and saved.audio[:4] == b"RIFF"
    assert _close(heard_en, "My dog is Pixel.") or _close(_last_word(heard_en), "Pixel")

    heard_fr, fr_stt = _heard(fr_wav, language="fr")
    renamed = run_audio_turn(
        store,
        brain,
        fr_wav,
        ears="faster-whisper",
        mouth="piper",
        language="fr",
        initial_prompt="Biscuit.",
    )
    assert renamed.status == "ok", renamed.reason
    dog = store.slot("fact:dog-name")
    assert dog is not None and _close(dog.value, "Biscuit"), dog.value if dog else None
    assert _close(_last_word(heard_fr), "Biscuit"), heard_fr
    assert en_stt < 30 and fr_stt < 30

    asked = run_turn(store, brain, "What's my dog's name?")
    assert dog.value in asked.reply
    reopened = MemoryStore(tmp_path / "home")
    assert _close(reopened.slot("fact:dog-name").value, "Biscuit")

    wav = tmp_path / "in.wav"
    wav.write_bytes(en_wav)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyper_ambient_maxi",
            "--home",
            str(tmp_path / "cli"),
            "--voice-file",
            str(wav),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cli_dog = MemoryStore(tmp_path / "cli").slot("fact:dog-name")
    assert cli_dog is not None and _close(cli_dog.value, "Pixel"), cli_dog.value if cli_dog else None
    reply = wav.with_name("in.reply.wav")
    assert reply.is_file() and reply.stat().st_size > 44


def _close(actual: str, expected: str) -> bool:
    left = "".join(ch for ch in actual.lower() if ch.isalpha())
    right = "".join(ch for ch in expected.lower() if ch.isalpha())
    if abs(len(left) - len(right)) > 2:
        return False
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for inner, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[inner] + 1,
                    current[inner - 1] + 1,
                    previous[inner - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1] <= 2


def _last_word(text: str) -> str:
    words = [word.strip(".,!?") for word in text.split() if word.strip(".,!?")]
    return words[-1] if words else text


def _spoken(text: str) -> tuple[bytes, float]:
    audio, status, reason = synthesize(text)
    assert status == "ok" and audio, reason
    # Warm call. The figure in the README is this pass, not the load.
    started = time.perf_counter()
    audio, status, reason = synthesize(text)
    warm = time.perf_counter() - started
    assert status == "ok" and audio, reason
    return audio, warm


def _heard(wav: bytes, language: str = "") -> tuple[str, float]:
    recognize(wav, language=language)
    started = time.perf_counter()
    text, status, reason = recognize(wav, language=language)
    elapsed = time.perf_counter() - started
    assert status == "ok" and text, reason
    return text, elapsed

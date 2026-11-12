"""Voice wiring that can run here, and the model card.

Anything that needs a GPU, a microphone, or model weights asserts the
NOT_RUN status instead of pretending the backend ran.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.config import load_models
from hyper_ambient_maxi.ears.wavutil import (
    SAMPLE_RATE,
    is_junk,
    pcm_from_wav,
    resample_to_16k,
    rms,
    wav_bytes,
)
from hyper_ambient_maxi.memory.store import MemoryStore
from hyper_ambient_maxi.selftest import run_selftest
from hyper_ambient_maxi.cli import _audio_file
from hyper_ambient_maxi.voice import AudioTurn, listen_status, run_audio_turn, speak

ROOT = Path(__file__).resolve().parents[1]


def _tone(amplitude: int = 8000, count: int = SAMPLE_RATE) -> bytes:
    samples = array("h")
    for index in range(count):
        samples.append(int(amplitude * math.sin(2 * math.pi * 220 * index / SAMPLE_RATE)))
    return wav_bytes(samples.tobytes())


def _silence() -> bytes:
    return wav_bytes(b"\x00\x00" * SAMPLE_RATE)


def test_wav_helpers_roundtrip_and_junk_filter():
    pcm, rate = pcm_from_wav(_tone())
    assert rate == SAMPLE_RATE
    assert rms(pcm) > 150
    assert rms(pcm_from_wav(_silence())[0]) == 0
    stretched = resample_to_16k(b"\x00\x01" * 8, 8000)
    assert len(stretched) == 32
    assert is_junk("thanks for watching")
    assert is_junk("no no no no")
    assert not is_junk("My dog is called Pixel.")


def test_loud_transcript_is_stored_and_silence_or_junk_is_not(tmp_path):
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    saved = run_audio_turn(
        store,
        brain,
        _tone(),
        transcript="My dog is called Pixel.",
        ears="text",
        mouth="text",
    )
    assert saved.status == "ok"
    assert saved.turn is not None
    assert saved.turn.memory_ok
    assert store.slot("fact:dog-name").value == "Pixel"

    junk = run_audio_turn(
        store,
        brain,
        _tone(),
        transcript="thanks for watching",
    )
    assert junk.status == "junk"
    assert junk.turn is None

    quiet = run_audio_turn(
        store,
        brain,
        _silence(),
        transcript="I just got a job at Airbus.",
    )
    assert quiet.status == "silence"
    assert store.slot("job:employer") is None


def test_speech_backends_without_weights_are_not_run(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    brain = ScriptedBrain()
    monkeypatch.setenv("HAM_VOICE_DIR", str(tmp_path / "no-voices"))
    for ears in ("faster-whisper", "lfm3-audio", "http-stt"):
        outcome = run_audio_turn(store, brain, _tone(), ears=ears, ears_url="")
        assert outcome.status == "NOT_RUN", ears
        assert outcome.reason.startswith("NOT_RUN")
        assert outcome.turn is None
    assert store.active() == []
    audio, status, reason = speak("hello", "lfm3-audio", "")
    assert audio is None
    assert status == "NOT_RUN"
    assert "LFM3-Audio-2B" in reason
    _, espeak_status, espeak_reason = speak("hello", "espeak", "")
    assert espeak_status == "NOT_RUN"
    assert espeak_reason.startswith("NOT_RUN")
    _, piper_status, piper_reason = speak("hello", "piper", "")
    assert piper_status == "NOT_RUN"
    assert piper_reason.startswith("NOT_RUN")
    listen, listen_reason = listen_status()
    assert listen == "NOT_RUN"
    assert "microphone" in listen_reason.lower() or "sounddevice" in listen_reason.lower()


def test_model_card_lists_local_gguf_and_marks_audio_not_run():
    models = load_models(ROOT / "config" / "models.toml")
    assert models.default_profile == "text"
    text = models.profile("text")
    assert text.run_status == "verified"
    assert text.requires_gpu is False
    assert text.brain == "scripted"
    brains = set(models.known_brains)
    assert brains == {
        "twil-8b",
        "lfm3-1-7b",
        "lfm2-5-8b-a1b",
        "nemotron-3-5-bolt",
        "vllm-d3-flash",
    }
    assert models.known_brains["lfm2-5-8b-a1b"]["alias"] == "lfm2.5-8b-a1b"
    audio = models.profile("lfm3-audio")
    assert audio.run_status == "NOT_RUN"
    assert audio.requires_gpu and audio.requires_audio_hardware
    assert audio.speech_model_id == "LiquidAI/LFM3-Audio-2B"
    assert audio.speech_dtype == "bfloat16"
    assert "English-only" in audio.description
    speech = models.known_speech["lfm3-audio-2b"]
    assert speech["languages"] == 18
    assert speech["run_status"] == "NOT_RUN"
    assert models.profile("thomas-4070").run_status == "NOT_RUN"
    assert models.profile("llama-local").brain_model == "lfm2.5-8b-a1b"
    assert models.profile("llama-local").brain_think is False
    assert models.profile("thomas-4070").brain_think is False
    assert models.profile("lfm3-audio").brain_think is False
    assert models.profile("llama-local").brain_max_sentences == 3
    assert models.profile("thomas-4070").brain_max_sentences == 3
    assert models.profile("text").brain_max_sentences == 3
    ears = models.profile("thomas-4070")
    assert ears.ears_model == "base"
    assert ears.ears_device == "cuda"
    assert ears.ears_language == ""
    assert ears.ears_hotwords == "Thomas"
    assert ears.ears_initial_prompt == ""


def test_hotwords_include_the_dog_and_people(tmp_path):
    from hyper_ambient_maxi.memory.store import ProposedFact
    from hyper_ambient_maxi.voice import hotwords_for

    store = MemoryStore(tmp_path)
    store.commit(
        [
            ProposedFact(
                key="fact:dog-name",
                type="personal_fact",
                summary="User's dog is called Biscuit",
                value="Biscuit",
                evidence="x",
            ),
            ProposedFact(
                key="fact:user-name",
                type="personal_fact",
                summary="User's name is Thomas",
                value="Thomas",
                evidence="x",
            ),
            ProposedFact(
                key="person:sister",
                type="personal_fact",
                summary="User's sister is Ada",
                value="Ada",
                evidence="x",
            ),
        ],
        source_agent="test",
        conversation_id="t",
    )
    assert hotwords_for(store, "Thomas Pixel") == "Thomas Pixel Biscuit Ada"
    assert hotwords_for(store, "Maxi") == "Biscuit Thomas Ada"


def test_voice_file_uses_the_profile_and_wav_writes_the_reply(tmp_path, monkeypatch):
    seen = {}

    def fake(store, brain, wav, **kwargs):
        del store, brain, wav
        seen.clear()
        seen.update(kwargs)
        turn = SimpleNamespace(reply="Your dog's name is Pixel.", memory_ok=True)
        return AudioTurn("ok", "http-tts", turn, b"RIFF-reply")

    monkeypatch.setattr("hyper_ambient_maxi.cli.run_audio_turn", fake)
    models = load_models(ROOT / "config" / "models.toml")
    src = tmp_path / "in.wav"
    src.write_bytes(b"RIFF-in")
    chosen = tmp_path / "chosen.wav"
    code = _audio_file(
        tmp_path / "home",
        models.profile("thomas-4070"),
        src,
        chosen,
        local_fallback=True,
    )
    assert code == 0
    assert seen["ears"] == "http-stt"
    assert seen["mouth"] == "http-tts"
    assert seen["ears_url"] == "http://127.0.0.1:8077"
    assert seen["mouth_url"] == "http://127.0.0.1:8078"
    assert seen["whisper_model"] == "base"
    assert seen["whisper_device"] == "cuda"
    assert seen["hotwords"] == "Thomas"
    assert seen["initial_prompt"] == ""
    assert seen["allow_download"] is False
    assert chosen.read_bytes() == b"RIFF-reply"

    code = _audio_file(
        tmp_path / "text-home",
        models.profile("text"),
        src,
        None,
        local_fallback=True,
    )
    assert code == 0
    assert seen["ears"] == "faster-whisper"
    assert seen["mouth"] == "piper"
    assert (tmp_path / "in.reply.wav").read_bytes() == b"RIFF-reply"

    code = _audio_file(
        tmp_path / "wav-home",
        models.profile("thomas-4070"),
        src,
        None,
        local_fallback=False,
    )
    assert code == 0
    assert seen["ears"] == "http-stt"
    assert (tmp_path / "in.reply.wav").read_bytes() == b"RIFF-reply"


def test_selftest_and_cli(tmp_path):
    assert run_selftest(tmp_path / "self") == 0
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    show = subprocess.run(
        [sys.executable, "-m", "hyper_ambient_maxi", "--show-models"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "NOT_RUN" in show.stdout
    assert "LiquidAI/LFM3-Audio-2B" in show.stdout
    assert "TwIL-8B-Q4_K_M.gguf" in show.stdout
    once = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyper_ambient_maxi",
            "--home",
            str(tmp_path / "cli"),
            "--once",
            "My dog is called Pixel.",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Pixel" in once.stdout
    assert "verified" in once.stderr
    asked = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyper_ambient_maxi",
            "--home",
            str(tmp_path / "cli"),
            "--once",
            "What's my dog's name?",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert asked.stdout.strip() == "Your dog's name is Pixel."
    wav = tmp_path / "tone.wav"
    wav.write_bytes(_tone())
    missing = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyper_ambient_maxi",
            "--home",
            str(tmp_path / "wav"),
            "--wav",
            str(wav),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 3
    assert missing.stdout.startswith("NOT_RUN")

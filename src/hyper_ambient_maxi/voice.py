"""Voice in and voice out, with an explicit NOT_RUN when they cannot run.

Text mode is the path the core tests execute. A WAV file can enter the
same turn when it has speech energy. faster-whisper and Piper run on
CPU when their files are in the local cache. A missing package, a
missing voice, or a missing model is ``NOT_RUN`` and writes nothing.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass

from hyper_ambient_maxi.ears.vad import trim_speech
from hyper_ambient_maxi.ears.wavutil import SILENCE_RMS, is_junk, pcm_from_wav, rms
from hyper_ambient_maxi.loop import TurnResult, run_turn
from hyper_ambient_maxi.memory.store import MemoryStore

_NOT_RUN_EARS = {
    "lfm3-audio": (
        "NOT_RUN: LiquidAI/LFM3-Audio-2B (BF16, 16 kHz in, 24 kHz out) "
        "is not loaded. Weights stay on the RTX 4070 under the LFM Open "
        "License. Memory writes do not go through this model."
    ),
}


@dataclass
class AudioTurn:
    status: str
    reason: str
    turn: TurnResult | None = None
    audio: bytes | None = None


def hotwords_for(store: MemoryStore, configured: str = "") -> str:
    """Profile hints plus the dog's name and people stored in memory.

    ``Maxi`` is left out. That name is the companion, not a person to hear.
    """
    names: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        text = " ".join(str(value).split())
        if not text:
            return
        for token in text.replace(",", " ").split():
            key = token.casefold()
            if not key or key == "maxi" or key in seen:
                continue
            seen.add(key)
            names.append(token)

    add(configured)
    dog = store.slot("fact:dog-name")
    if dog is not None:
        add(dog.value)
    user = store.slot("fact:user-name")
    if user is not None:
        add(user.value)
    for fact in store.active():
        if fact.key.startswith("person:") or fact.key.startswith("people:"):
            add(fact.value)
    return " ".join(names)


def run_audio_turn(
    store: MemoryStore,
    brain,
    wav: bytes,
    *,
    ears: str = "text",
    mouth: str = "text",
    ears_url: str = "",
    mouth_url: str = "",
    transcript: str | None = None,
    conversation_id: str = "maxi",
    source_agent: str = "maxi-voice",
    allow_download: bool = False,
    language: str = "",
    initial_prompt: str = "",
    hotwords: str = "",
    whisper_model: str = "",
    whisper_device: str = "",
) -> AudioTurn:
    try:
        pcm, _rate = pcm_from_wav(wav)
    except Exception as exc:
        return AudioTurn("error", f"unreadable wav: {exc}")
    spoken = trim_speech(pcm)
    if spoken is None or rms(spoken) < SILENCE_RMS:
        return AudioTurn("silence", "no speech region; nothing stored")

    text = transcript
    if text is None:
        language, initial_prompt, hotwords = _stt_hints(
            store, language, initial_prompt, hotwords, whisper_model
        )
        text, status, reason = transcribe(
            wav,
            ears,
            ears_url,
            allow_download=allow_download,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
        )
        if status != "ok" or not text:
            return AudioTurn(status, reason)
    if is_junk(text):
        return AudioTurn("junk", "transcript rejected; nothing stored")

    turn = run_turn(
        store,
        brain,
        text,
        conversation_id=conversation_id,
        source_agent=source_agent,
    )
    turn.voice_status = "audio"
    audio, speech_status, speech_reason = speak(
        turn.reply,
        mouth,
        mouth_url,
        language=_reply_language(text, turn.reply, store, language),
    )
    # A missing speaker does not roll back a verified write. The caller
    # still has the text reply; the reason says the voice was NOT_RUN.
    if speech_status == "NOT_RUN":
        return AudioTurn("ok", speech_reason, turn, None)
    return AudioTurn("ok", speech_reason or speech_status, turn, audio)


def transcribe(
    wav: bytes,
    backend: str,
    url: str,
    *,
    allow_download: bool = False,
    language: str = "",
    initial_prompt: str = "",
    hotwords: str = "",
    whisper_model: str = "",
    whisper_device: str = "",
) -> tuple[str | None, str, str]:
    if backend in {"text", "scripted", ""}:
        return None, "NOT_RUN", (
            "NOT_RUN: this profile has no speech recognizer. "
            "Use --text or --once."
        )
    if backend == "http-stt":
        if not url:
            return None, "NOT_RUN", "NOT_RUN: http-stt has no base_url"
        try:
            return (
                _post_transcription(
                    url,
                    wav,
                    language=language,
                    initial_prompt=initial_prompt,
                    hotwords=hotwords,
                ),
                "ok",
                "",
            )
        except Exception as exc:
            return None, "NOT_RUN", f"NOT_RUN: stt server unreachable ({exc})"
    if backend == "faster-whisper":
        from hyper_ambient_maxi.ears.local_speech import recognize

        return recognize(
            wav,
            allow_download=allow_download,
            model=whisper_model,
            device=whisper_device,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
    if backend in _NOT_RUN_EARS:
        return None, "NOT_RUN", _NOT_RUN_EARS[backend]
    return None, "NOT_RUN", f"NOT_RUN: unknown ears backend {backend}"


def speak(text: str, backend: str, url: str, *, language: str = "") -> tuple[bytes | None, str, str]:
    if backend in {"text", "scripted", ""}:
        return None, "text", "text"
    if backend == "espeak":
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            return None, "NOT_RUN", (
                "NOT_RUN: espeak is installed but playback needs an audio "
                "device, which was not exercised."
            )
        return None, "NOT_RUN", "NOT_RUN: espeak-ng is not installed"
    if backend == "http-tts":
        if not url:
            return None, "NOT_RUN", "NOT_RUN: http-tts has no base_url"
        try:
            return _post_speech(url, text, language=language), "ok", "http-tts"
        except Exception as exc:
            return None, "NOT_RUN", f"NOT_RUN: tts server unreachable ({exc})"
    if backend == "lfm3-audio":
        return None, "NOT_RUN", _NOT_RUN_EARS["lfm3-audio"]
    if backend == "piper":
        from hyper_ambient_maxi.ears.local_speech import synthesize

        return synthesize(text, language=language)
    return None, "NOT_RUN", f"NOT_RUN: unknown mouth backend {backend}"


def listen_status() -> tuple[str, str]:
    """Report whether capture could start. Does not open a microphone."""
    try:
        import sounddevice  # type: ignore
    except Exception:
        return "NOT_RUN", (
            "NOT_RUN: sounddevice is not installed, and this process does "
            "not open a microphone. Use --text."
        )
    try:
        devices = sounddevice.query_devices()
    except Exception as exc:
        return "NOT_RUN", f"NOT_RUN: audio devices could not be queried ({exc})"
    if not devices:
        return "NOT_RUN", "NOT_RUN: no audio devices"
    return "NOT_RUN", (
        "NOT_RUN: an audio library imported, but capture was not started. "
        "Maxi does not record in v0."
    )


def _post_transcription(
    base_url: str,
    wav: bytes,
    *,
    language: str = "",
    initial_prompt: str = "",
    hotwords: str = "",
) -> str:
    url = base_url.rstrip("/") + "/v1/audio/transcriptions"
    boundary = "----hyperambientmaxi"
    parts = [
        _file_part(boundary, "file", "turn.wav", "audio/wav", wav),
    ]
    for name, value in (
        ("language", language),
        ("initial_prompt", initial_prompt),
        ("hotwords", hotwords),
    ):
        if value.strip():
            parts.append(_text_part(boundary, name, value.strip()))
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = str(payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("empty transcription")
    return text


def _file_part(boundary: str, name: str, filename: str, content_type: str, data: bytes) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + b"\r\n"


def _text_part(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode()


def _post_speech(base_url: str, text: str, *, language: str = "") -> bytes:
    url = base_url.rstrip("/") + "/v1/audio/speech"
    payload = {"input": text, "voice": "female"}
    if language.strip():
        payload["language"] = language.strip()
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    if not audio:
        raise RuntimeError("empty audio")
    return audio


# Seed words for first-class slots. Whisper ``base`` heard "My dog is
# Pixel." as "My door is pixel." until "dog" was in the hotwords.
# "chien" similarly steadies "s'appelle mon chien".
_STT_SEED = "dog chien"


def _stt_hints(
    store: MemoryStore,
    language: str,
    initial_prompt: str,
    hotwords: str,
    whisper_model: str,
) -> tuple[str, str, str]:
    """Fill a language preference, seed hotwords, and a short name prompt.

    A name prompt helped Whisper ``base`` hear Léa and Biscuit. The same
    prompt made ``tiny`` emit garbage on one English question, so it is
    only added when the model is not tiny. Seed hotwords are always
    merged; they fixed "dog" vs "door" on the first turn, before any
    name is stored.
    """
    from hyper_ambient_maxi.ears.local_speech import whisper_model_name

    names = hotwords_for(store, f"{_STT_SEED} {hotwords}")
    if not language.strip():
        for key in ("fact:language", "fact:preferred-language"):
            fact = store.slot(key)
            if fact is not None and str(fact.value).strip():
                language = str(fact.value).strip()
                break
    if not initial_prompt.strip() and names:
        model = (whisper_model or whisper_model_name()).strip() or whisper_model_name()
        if model.lower() != "tiny":
            initial_prompt = ". ".join(names.split()) + "."
    hotwords = names or hotwords
    return language, initial_prompt, hotwords

def _reply_language(user_text: str, reply: str, store: MemoryStore, configured: str = "") -> str:
    """Prefer the reply's language, then the user's turn, then a stored preference."""
    from hyper_ambient_maxi.ears.local_speech import speech_language

    preferred = (configured or "").strip()
    if not preferred:
        for key in ("fact:language", "fact:preferred-language"):
            fact = store.slot(key)
            if fact is not None and str(fact.value).strip():
                preferred = str(fact.value).strip()
                break
    if not preferred:
        preferred = speech_language(user_text)
    return speech_language(reply, preferred=preferred)

"""CPU speech, used only when the local files are already on disk.

faster-whisper and Piper are optional. A missing package or a missing
voice file is ``NOT_RUN``. Nothing here opens a microphone. Weights are
fetched only by ``--download-models``, never by a turn or a server.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import unicodedata
import urllib.request
import wave
from pathlib import Path

DOWNLOAD_HINT = "Run python -m hyper_ambient_maxi --download-models"
WHISPER_SETUP_MODEL = "base"
# Hugging Face rhasspy/piper-voices v1.0.0. The json sits beside the onnx.
PIPER_FILES = (
    (
        "en_US-lessac-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx",
    ),
    (
        "en_US-lessac-low.onnx.json",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
    ),
    (
        "fr_FR-siwis-low.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx",
    ),
    (
        "fr_FR-siwis-low.onnx.json",
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/fr/fr_FR/siwis/low/fr_FR-siwis-low.onnx.json",
    ),
)

_WHISPER = None
_WHISPER_NAME = ""
_PIPER: dict[str, object] = {}

_FRENCH = re.compile(
    r"[àâçéèêëîïôùûüœ]|(?:\b(?:je|tu|il|elle|mon|ma|mes|renommé|renomme)\b)|l['’]ai",
    re.I,
)


def voice_cache() -> Path:
    override = os.environ.get("HAM_VOICE_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "hyper-ambient-maxi"


def whisper_model_name() -> str:
    return os.environ.get("HAM_WHISPER_MODEL", "tiny").strip() or "tiny"


def faster_whisper_ready() -> bool:
    if importlib.util.find_spec("faster_whisper") is None:
        return False
    return _whisper_snapshot() is not None


def piper_ready() -> bool:
    if importlib.util.find_spec("piper") is None:
        return False
    voices = piper_voices()
    return "en" in voices and "fr" in voices


def faster_whisper_status() -> tuple[str, str]:
    if importlib.util.find_spec("faster_whisper") is None:
        return "NOT_RUN", "NOT_RUN: faster-whisper is not installed (pip install -e '.[voice]')"
    snapshot = _whisper_snapshot()
    if snapshot is None:
        name = whisper_model_name()
        return (
            "NOT_RUN",
            "NOT_RUN: faster-whisper "
            f"{name} is not in {voice_cache() / 'whisper'}. {DOWNLOAD_HINT}",
        )
    return "ok", f"{whisper_model_name()} int8 cpu"


def piper_status() -> tuple[str, str]:
    if importlib.util.find_spec("piper") is None:
        return "NOT_RUN", "NOT_RUN: piper-tts is not installed (pip install -e '.[voice]')"
    voices = piper_voices()
    if "en" not in voices or "fr" not in voices:
        return (
            "NOT_RUN",
            "NOT_RUN: Piper needs en_US-lessac-low and fr_FR-siwis-low in "
            f"{voice_cache() / 'piper'}. {DOWNLOAD_HINT}",
        )
    names = ", ".join(path.stem for path in (voices["en"], voices["fr"]))
    return "ok", names


def piper_voices() -> dict[str, Path]:
    """Small ONNX voices. English and French are chosen by filename."""
    root = voice_cache() / "piper"
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*.onnx")):
        name = path.name.lower()
        if name.startswith("en") and "en" not in found:
            found["en"] = path
        elif name.startswith("fr") and "fr" not in found:
            found["fr"] = path
    return found


def speech_text(text: str) -> str:
    """NFC. Piper's French map has é and misses the combining tilde in "renommé"."""
    return unicodedata.normalize("NFC", text or "")


def synthesize(text: str) -> tuple[bytes | None, str, str]:
    status, reason = piper_status()
    if status != "ok":
        return None, status, reason
    spoken = speech_text(text)
    voice = _piper_voice(_voice_key(spoken))
    if voice is None:
        return None, "NOT_RUN", reason
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        voice.synthesize_wav(spoken, handle)  # type: ignore[attr-defined]
    audio = buffer.getvalue()
    if not audio:
        return None, "NOT_RUN", "NOT_RUN: Piper returned empty audio"
    return audio, "ok", "piper"


def transcribe_options(
    *,
    language: str = "",
    initial_prompt: str = "",
    hotwords: str = "",
) -> dict:
    """Decoding settings. beam 5 and temperature 0 stay fixed.

    ``language``, ``initial_prompt``, and ``hotwords`` are omitted when
    empty so an unknown language is still detected.
    """
    options: dict = {
        "vad_filter": False,
        "temperature": 0.0,
        "beam_size": 5,
        "condition_on_previous_text": False,
    }
    if language.strip():
        options["language"] = language.strip()
    if initial_prompt.strip():
        options["initial_prompt"] = initial_prompt.strip()
    if hotwords.strip():
        options["hotwords"] = hotwords.strip()
    return options


def recognize(
    wav: bytes,
    *,
    allow_download: bool = False,
    model: str = "",
    device: str = "",
    language: str = "",
    initial_prompt: str = "",
    hotwords: str = "",
) -> tuple[str | None, str, str]:
    if importlib.util.find_spec("faster_whisper") is None:
        return (
            None,
            "NOT_RUN",
            "NOT_RUN: faster-whisper is not installed (pip install -e '.[voice]')",
        )
    name = model.strip() or whisper_model_name()
    if _whisper_snapshot(name) is None and not allow_download:
        return (
            None,
            "NOT_RUN",
            "NOT_RUN: faster-whisper "
            f"{name} is not in {voice_cache() / 'whisper'}. {DOWNLOAD_HINT}",
        )
    try:
        audio = _float_audio(wav)
        loaded = _whisper_model(name, device, allow_download=allow_download)
        segments, _info = loaded.transcribe(
            audio,
            **transcribe_options(
                language=language,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
            ),
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        return None, "NOT_RUN", f"NOT_RUN: faster-whisper failed ({exc})"
    if not text:
        return None, "NOT_RUN", "NOT_RUN: faster-whisper returned an empty transcript"
    return text, "ok", "faster-whisper"


def _voice_key(text: str) -> str:
    if _FRENCH.search(text or ""):
        return "fr"
    return "en"


def resolve_device(requested: str) -> str:
    """``cuda`` when the profile asks for it and a device is present."""
    want = (requested or "cpu").strip().lower()
    if want != "cuda":
        return "cpu"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"


def _whisper_snapshot(name: str | None = None) -> Path | None:
    root = voice_cache() / "whisper"
    if not root.is_dir():
        return None
    needle = f"faster-whisper-{name or whisper_model_name()}"
    for path in root.rglob("model.bin"):
        if needle in str(path):
            return path
    return None


def _whisper_model(name: str, device: str, allow_download: bool):
    global _WHISPER, _WHISPER_NAME
    chosen = resolve_device(device)
    key = f"{name}:{chosen}"
    if _WHISPER is not None and _WHISPER_NAME == key:
        return _WHISPER
    if _whisper_snapshot(name) is None and not allow_download:
        raise RuntimeError(f"{name} is not cached")
    from faster_whisper import WhisperModel

    # One CPU thread keeps a short name stable. Several threads on this
    # Xeon changed "Pixel" and "Biscuit" between calls.
    kwargs = {
        "device": chosen,
        "compute_type": "float16" if chosen == "cuda" else "int8",
        "download_root": str(voice_cache() / "whisper"),
    }
    if chosen == "cpu":
        kwargs["cpu_threads"] = 1
    _WHISPER = WhisperModel(name, **kwargs)
    _WHISPER_NAME = key
    return _WHISPER


def _piper_voice(key: str):
    voices = piper_voices()
    path = voices.get(key) or voices.get("en") or voices.get("fr")
    if path is None:
        return None
    cached = _PIPER.get(str(path))
    if cached is not None:
        return cached
    from piper import PiperVoice

    voice = PiperVoice.load(path, use_cuda=False)
    _PIPER[str(path)] = voice
    return voice


def download_voice_models() -> list[str]:
    """Fetch Whisper base and the Piper voices. Skip a file that is already cached.

    Called only from ``--download-models``. A turn and ``--serve-ears`` do not.
    """
    lines = [_ensure_whisper(WHISPER_SETUP_MODEL)]
    root = voice_cache() / "piper"
    root.mkdir(parents=True, exist_ok=True)
    for filename, url in PIPER_FILES:
        dest = root / filename
        if dest.is_file() and dest.stat().st_size > 0:
            lines.append(f"cached {filename}")
            continue
        _fetch(url, dest)
        lines.append(f"downloaded {filename}")
    return lines


def _ensure_whisper(name: str) -> str:
    if _whisper_snapshot(name) is not None:
        return f"cached faster-whisper {name}"
    if importlib.util.find_spec("faster_whisper") is None:
        raise RuntimeError("faster-whisper is not installed (pip install -e '.[voice]')")
    from faster_whisper import WhisperModel

    WhisperModel(
        name,
        device="cpu",
        compute_type="int8",
        download_root=str(voice_cache() / "whisper"),
        cpu_threads=1,
    )
    if _whisper_snapshot(name) is None:
        raise RuntimeError(f"faster-whisper {name} did not land in the cache")
    return f"downloaded faster-whisper {name}"


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "hyper-ambient-maxi"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"empty download {url}")
    temporary = dest.with_name(dest.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(dest)


def _float_audio(wav: bytes):
    """16 kHz float32. Passing an array avoids PyAV's file opener."""
    import numpy as np

    from hyper_ambient_maxi.ears.wavutil import pcm_from_wav

    pcm, _rate = pcm_from_wav(wav)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(audio)

"""HTTP ears and mouth, the same contract as thomas-4070.

The multipart parser always runs. The live servers are skipped when the
Whisper or Piper cache is absent.
"""

from __future__ import annotations

import threading

import pytest

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.ears.http_speech import (
    bind_speech,
    form_fields,
    serve,
    wav_from_multipart,
)
from hyper_ambient_maxi.ears.local_speech import (
    faster_whisper_ready,
    piper_ready,
    resolve_device,
    synthesize,
    transcribe_options,
)
from hyper_ambient_maxi.loop import run_turn
from hyper_ambient_maxi.memory.store import MemoryStore
from hyper_ambient_maxi.voice import run_audio_turn

# Used only to check that form fields and decode options are forwarded.
_PROMPT = "Pixel. Luna."
# Per-turn prompts name only the word spoken on that turn. A shared
# prompt that also contains the previous name biased French STT.
EN = "My dog is Pixel."
FR = "Je l'ai renommé Luna."


def names_close(actual: str, expected: str) -> bool:
    """Tiny sometimes hears Pexal for Pixel. Distance 2 still counts."""
    return _edit_distance(actual.lower(), expected.lower()) <= 2


def _edit_distance(left: str, right: str) -> int:
    if abs(len(left) - len(right)) > 2:
        return abs(len(left) - len(right))
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
    return previous[-1]


def test_transcription_form_keeps_language_and_names():
    from hyper_ambient_maxi.voice import _file_part, _text_part

    boundary = "----hyperambientmaxi"
    wav = b"RIFF-pixel"
    content_type = f"multipart/form-data; boundary={boundary}"
    body = b"".join(
        [
            _file_part(boundary, "file", "turn.wav", "audio/wav", wav),
            _text_part(boundary, "language", "fr"),
            _text_part(boundary, "initial_prompt", _PROMPT),
            _text_part(boundary, "hotwords", "Pixel Biscuit"),
        ]
    ) + f"--{boundary}--\r\n".encode()
    fields = form_fields(body, content_type)
    assert fields["file"] == wav
    assert fields["language"] == b"fr"
    assert fields["initial_prompt"] == _PROMPT.encode()
    assert fields["hotwords"] == b"Pixel Biscuit"
    assert wav_from_multipart(body, content_type) == wav


def test_near_name_accepts_pexal_and_decode_stays_greedy():
    assert names_close("Pexal", "Pixel")
    assert names_close("Pixel", "Pixel")
    assert not names_close("Biscuit", "Pixel")
    plain = transcribe_options()
    assert plain["temperature"] == 0.0
    assert plain["beam_size"] == 5
    assert plain["condition_on_previous_text"] is False
    assert "language" not in plain
    hinted = transcribe_options(language="fr", initial_prompt=_PROMPT, hotwords="Biscuit")
    assert hinted["language"] == "fr"
    assert hinted["initial_prompt"] == _PROMPT
    assert hinted["beam_size"] == 5
    assert hinted["temperature"] == 0.0
    assert resolve_device("") == "cpu"
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") in {"cpu", "cuda"}


def test_multipart_file_matches_the_voice_client():
    wav = b"RIFF" + b"\x00\x01" * 32
    boundary = "----hyperambientmaxi"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="turn.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    content_type = f"multipart/form-data; boundary={boundary}"
    assert wav_from_multipart(body, content_type) == wav


def test_serve_without_a_cache_does_not_listen(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HAM_VOICE_DIR", str(tmp_path / "empty"))
    assert serve("ears") == 3
    assert serve("mouth") == 3
    captured = capsys.readouterr()
    assert "NOT_RUN" in captured.out
    assert "--download-models" in captured.out


@pytest.mark.skipif(
    not (faster_whisper_ready() and piper_ready()),
    reason="faster-whisper or Piper voices are not in the local cache",
)
def test_http_ears_and_mouth_round_trip(tmp_path):
    ears = bind_speech("ears")
    mouth = bind_speech("mouth")
    for server in (ears, mouth):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    ears_url = f"http://127.0.0.1:{ears.server_address[1]}"
    mouth_url = f"http://127.0.0.1:{mouth.server_address[1]}"
    try:
        en, status, reason = synthesize(EN, language="en")
        assert status == "ok" and en, reason
        fr, status, reason = synthesize(FR, language="fr")
        assert status == "ok" and fr, reason
        store = MemoryStore(tmp_path)
        brain = ScriptedBrain()
        named = run_audio_turn(
            store,
            brain,
            en,
            ears="http-stt",
            mouth="http-tts",
            ears_url=ears_url,
            mouth_url=mouth_url,
            language="en",
            initial_prompt="Pixel.",
            hotwords="Pixel",
        )
        assert named.status == "ok", named.reason
        assert named.turn is not None and named.turn.memory_ok
        dog = store.slot("fact:dog-name")
        assert dog is not None and names_close(dog.value, "Pixel"), dog.value if dog else None
        assert named.audio and named.audio[:4] == b"RIFF"
        renamed = run_audio_turn(
            store,
            brain,
            fr,
            ears="http-stt",
            mouth="http-tts",
            ears_url=ears_url,
            mouth_url=mouth_url,
            language="fr",
            initial_prompt="Luna.",
            hotwords="Luna",
        )
        assert renamed.status == "ok", renamed.reason
        dog = store.slot("fact:dog-name")
        assert dog is not None and names_close(dog.value, "Luna"), dog.value if dog else None
        assert not names_close(dog.value, "Pixel")
        assert renamed.audio and renamed.audio[:4] == b"RIFF"
        asked = run_turn(store, brain, "What's my dog's name?")
        assert dog.value in asked.reply
        assert "Pixel" not in asked.reply
    finally:
        ears.shutdown()
        mouth.shutdown()
        ears.server_close()
        mouth.server_close()


def test_each_request_logs_method_path_status_and_duration(capsys):
    import re
    import urllib.request

    server = bind_speech("ears")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
    err = capsys.readouterr().err
    assert re.search(r"GET /health 200 \d+ms", err)


def test_download_is_explicit_and_skips_files_already_cached(tmp_path, monkeypatch, capsys):
    from hyper_ambient_maxi.cli import main
    from hyper_ambient_maxi.ears import local_speech

    monkeypatch.setenv("HAM_VOICE_DIR", str(tmp_path / "voices"))
    fetched: list[tuple[str, str]] = []

    def fake_whisper(name: str) -> str:
        fetched.append(("whisper", name))
        return f"downloaded faster-whisper {name}"

    def fake_fetch(url: str, dest) -> None:
        fetched.append((url, dest.name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"onnx")

    real_whisper = local_speech._ensure_whisper
    monkeypatch.setattr(local_speech, "_ensure_whisper", fake_whisper)
    monkeypatch.setattr(local_speech, "_fetch", fake_fetch)
    first = local_speech.download_voice_models()
    assert ("whisper", "base") in fetched
    names = {name for kind, name in fetched if kind != "whisper"}
    assert "en_US-lessac-low.onnx" in names
    assert "en_US-lessac-low.onnx.json" in names
    assert "fr_FR-siwis-low.onnx" in names
    assert "fr_FR-siwis-low.onnx.json" in names
    assert any(line.startswith("downloaded en_US-lessac-low.onnx") for line in first)

    snapshot = (
        tmp_path
        / "voices"
        / "whisper"
        / "models--Systran--faster-whisper-base"
        / "snapshots"
        / "x"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"x")
    fetched.clear()
    monkeypatch.setattr(local_speech, "_ensure_whisper", real_whisper)
    second = local_speech.download_voice_models()
    assert fetched == []
    assert second[0] == "cached faster-whisper base"
    assert any(line == "cached en_US-lessac-low.onnx" for line in second)

    monkeypatch.setattr(local_speech, "download_voice_models", lambda: ["cached faster-whisper base"])
    assert main(["--download-models"]) == 0
    assert "cached faster-whisper base" in capsys.readouterr().out


def test_piper_folds_a_combining_tilde_before_synthesis(monkeypatch):
    from hyper_ambient_maxi.ears import local_speech

    seen: dict[str, str] = {}

    class Voice:
        def synthesize_wav(self, text: str, handle) -> None:
            seen["text"] = text
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00")

    monkeypatch.setattr(local_speech, "piper_status", lambda: ("ok", "voices"))
    monkeypatch.setattr(local_speech, "_piper_voice", lambda key: Voice())
    raw = "Je l'ai renomme\u0301 Biscuit."
    audio, status, reason = local_speech.synthesize(raw)
    assert status == "ok", reason
    assert audio and audio[:4] == b"RIFF"
    assert seen["text"] == "Je l'ai renommé Biscuit."
    assert "\u0303" not in seen["text"]

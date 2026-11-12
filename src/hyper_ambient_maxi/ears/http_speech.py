"""Loopback speech servers for the thomas-4070 HTTP contract.

Ears:  ``POST /v1/audio/transcriptions`` multipart field ``file``
       → ``{"text": "..."}``.
Mouth: ``POST /v1/audio/speech`` JSON ``{"input", "voice"}``
       → WAV bytes.

Both are faster-whisper and Piper. No microphone, no LFM3. A missing
cache prints ``NOT_RUN`` and does not listen.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hyper_ambient_maxi.ears.local_speech import (
    DOWNLOAD_HINT,
    piper_status,
    recognize,
    synthesize,
)

_LOCK = threading.Lock()


class SpeechServer(ThreadingHTTPServer):
    allow_reuse_address = True
    kind = "ears"
    model_name = ""
    language = ""
    device = ""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # send_response logs via this hook. The timed line is written below.
        return

    def do_GET(self) -> None:  # noqa: N802
        started = time.perf_counter()
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        self._status = 500
        try:
            if path in {"/", "/health"}:
                self._json(200, {"ok": True, "kind": self.server.kind})
                return
            self._json(404, {"error": "not found"})
        finally:
            access_log("GET", path, self._status, time.perf_counter() - started)

    def do_POST(self) -> None:  # noqa: N802
        started = time.perf_counter()
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        self._status = 500
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            kind = self.server.kind
            try:
                if kind == "ears" and path.endswith("/v1/audio/transcriptions"):
                    self._transcribe(body, self.headers.get("Content-Type", ""))
                    return
                if kind == "mouth" and path.endswith("/v1/audio/speech"):
                    self._speak(body)
                    return
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(404, {"error": "not found"})
        finally:
            access_log("POST", path, self._status, time.perf_counter() - started)

    def _transcribe(self, body: bytes, content_type: str) -> None:
        fields = form_fields(body, content_type)
        wav = fields.get("file", b"")
        if not wav:
            raise ValueError("multipart body has no file field")
        language = _field(fields, "language") or self.server.language
        with _LOCK:
            text, status, reason = recognize(
                wav,
                model=self.server.model_name,
                device=self.server.device,
                language=language,
                initial_prompt=_field(fields, "initial_prompt"),
                hotwords=_field(fields, "hotwords"),
            )
        if status != "ok" or not text:
            self._json(503, {"error": reason or "empty transcript"})
            return
        self._json(200, {"text": text})

    def _speak(self, body: bytes) -> None:
        payload = json.loads(body.decode("utf-8"))
        text = str(payload.get("input") or "").strip()
        if not text:
            self._json(400, {"error": "empty input"})
            return
        language = str(payload.get("language") or "").strip()
        with _LOCK:
            audio, status, reason = synthesize(text, language=language)
        if status != "ok" or not audio:
            self._json(503, {"error": reason or "empty audio"})
            return
        self._bytes(200, audio, "audio/wav")

    def _json(self, code: int, payload: dict) -> None:
        self._bytes(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _bytes(self, code: int, data: bytes, content_type: str) -> None:
        self._status = code
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def access_log(method: str, path: str, status: int, seconds: float) -> None:
    """One line: method, path, status, duration."""
    sys.stderr.write(f"{method} {path} {status} {seconds * 1000:.0f}ms\n")


def form_fields(body: bytes, content_type: str) -> dict[str, bytes]:
    """Multipart parts keyed by form name. ``file`` is the WAV bytes."""
    lowered = content_type.lower()
    if "boundary=" not in lowered:
        raise ValueError("missing multipart boundary")
    token = content_type.split("boundary=", 1)[1].split(";", 1)[0].strip().strip('"')
    if not token:
        raise ValueError("empty multipart boundary")
    marker = b"--" + token.encode("utf-8")
    fields: dict[str, bytes] = {}
    for part in body.split(marker):
        chunk = part[2:] if part.startswith(b"\r\n") else part
        if b"\r\n\r\n" not in chunk:
            continue
        header, data = chunk.split(b"\r\n\r\n", 1)
        name = _part_name(header)
        if not name:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        fields[name] = data
    return fields


def wav_from_multipart(body: bytes, content_type: str) -> bytes:
    """The file part sent by ``voice._post_transcription``."""
    data = form_fields(body, content_type).get("file", b"")
    if not data:
        raise ValueError("multipart body has no file field")
    return data


def _part_name(header: bytes) -> str:
    text = header.decode("utf-8", errors="replace")
    for token in text.split(";"):
        piece = token.strip()
        if piece.startswith("name="):
            return piece.split("=", 1)[1].strip().strip('"')
    return ""


def _field(fields: dict[str, bytes], name: str) -> str:
    return fields.get(name, b"").decode("utf-8", errors="replace").strip()


def bind_speech(
    kind: str,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    model: str = "",
    language: str = "",
    device: str = "",
) -> SpeechServer:
    if kind not in {"ears", "mouth"}:
        raise ValueError(f"unknown speech server {kind}")
    server = SpeechServer((host, port), _Handler)
    server.kind = kind
    server.model_name = model.strip()
    server.language = language.strip()
    server.device = device.strip()
    return server


def serve(
    kind: str,
    host: str = "127.0.0.1",
    port: int = 8077,
    *,
    model: str = "",
    language: str = "",
    device: str = "",
) -> int:
    """Listen until interrupted. Missing weights exit 3 and do not bind."""
    if kind == "ears":
        from hyper_ambient_maxi.ears.local_speech import whisper_model_name, _whisper_snapshot

        name = model.strip() or whisper_model_name()
        if _whisper_snapshot(name) is None:
            print(
                "NOT_RUN: faster-whisper "
                f"{name} is not in the local cache. {DOWNLOAD_HINT}"
            )
            return 3
    else:
        status, reason = piper_status()
        if status != "ok":
            print(reason)
            return 3
    try:
        server = bind_speech(
            kind, host, port, model=model, language=language, device=device
        )
    except OSError as exc:
        print(f"NOT_RUN: {host}:{port} ({exc})")
        return 1
    bound = server.server_address
    print(f"{kind} http://{bound[0]}:{bound[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0

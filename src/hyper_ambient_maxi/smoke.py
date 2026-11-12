"""One pass over the pieces Thomas can run on the RTX 4070.

Each line is ``PASS``, ``NOT_RUN``, or ``FAIL``. A closed port, a missing
optional library, and a headless display are ``NOT_RUN``. The self-test
runs in a temporary home so this probe does not write the user's journal.
``FAIL`` is reserved for a self-test that does not pass.
"""

from __future__ import annotations

import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path

from hyper_ambient_maxi.ears.hotkey import open_hotkey
from hyper_ambient_maxi.presence_window import display_status
from hyper_ambient_maxi.selftest import run_selftest
from hyper_ambient_maxi.ears.local_speech import faster_whisper_status, piper_status
from hyper_ambient_maxi.voice import listen_status


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""

    def line(self) -> str:
        if self.detail:
            return f"{self.name} {self.status} {self.detail}"
        return f"{self.name} {self.status}"


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run_smoke() -> list[Check]:
    home = Path(tempfile.mkdtemp(prefix="maxi-smoke-"))
    code = run_selftest(home)
    checks = [
        Check("selftest", "PASS" if code == 0 else "FAIL", str(home)),
        _port("llama-server", "127.0.0.1", 8080),
        _port("ears-8077", "127.0.0.1", 8077),
        _port("mouth-8078", "127.0.0.1", 8078),
    ]
    hotkey = open_hotkey("pynput")
    checks.append(
        Check(
            "pynput",
            "PASS" if hotkey.status == "ok" else "NOT_RUN",
            "" if hotkey.status == "ok" else hotkey.reason,
        )
    )
    window, window_reason = display_status()
    checks.append(Check("presence-window", "PASS" if window == "ok" else "NOT_RUN", "" if window == "ok" else window_reason))
    mic, mic_reason = listen_status()
    checks.append(Check("microphone", "PASS" if mic == "ok" else "NOT_RUN", "" if mic == "ok" else mic_reason))
    checks.append(_optional("faster-whisper", faster_whisper_status()))
    checks.append(_optional("piper", piper_status()))
    return checks


def format_smoke(checks: list[Check]) -> str:
    return "\n".join(check.line() for check in checks)


def smoke_exit(checks: list[Check]) -> int:
    if any(check.status == "FAIL" for check in checks):
        return 1
    return 0


def _optional(name: str, result: tuple[str, str]) -> Check:
    status, detail = result
    if status == "ok":
        return Check(name, "PASS", detail)
    return Check(name, "NOT_RUN", detail)


def _port(name: str, host: str, port: int) -> Check:
    if port_open(host, port):
        return Check(name, "PASS", f"{host}:{port}")
    return Check(name, "NOT_RUN", f"{host}:{port} closed")

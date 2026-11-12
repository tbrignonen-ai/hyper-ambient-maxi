"""Smoke report: PASS, NOT_RUN, or FAIL. Closed ports stay NOT_RUN."""

import os
import socket
import subprocess
import sys
from pathlib import Path

from hyper_ambient_maxi.smoke import format_smoke, port_open, run_smoke, smoke_exit

ROOT = Path(__file__).resolve().parents[1]


def test_port_open_follows_a_local_listener():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert port_open("127.0.0.1", port, timeout=0.5) is True
    finally:
        sock.close()
    assert port_open("127.0.0.1", port, timeout=0.3) is False


def test_smoke_passes_memory_and_marks_missing_pieces():
    checks = run_smoke()
    text = format_smoke(checks)
    by_name = {check.name: check for check in checks}
    assert by_name["selftest"].status == "PASS"
    assert by_name["pynput"].status == "NOT_RUN"
    assert by_name["microphone"].status == "NOT_RUN"
    assert "selftest PASS" in text
    headless = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    if headless:
        assert by_name["presence-window"].status == "NOT_RUN"
    for name in ("llama-server", "ears-8077", "mouth-8078"):
        assert by_name[name].status in {"PASS", "NOT_RUN"}
    assert smoke_exit(checks) == 0
    assert "Start-Maxi.ps1" in (ROOT / "scripts" / "Start-Maxi.ps1").read_text(encoding="utf-8")
    assert "thomas-4070" in (ROOT / "scripts" / "Start-Maxi.ps1").read_text(encoding="utf-8")
    assert "--smoke" in (ROOT / "scripts" / "Smoke-4070.ps1").read_text(encoding="utf-8")


def test_cli_smoke_prints_selftest_pass():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "hyper_ambient_maxi", "--smoke"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "selftest PASS" in proc.stdout
    assert "pynput NOT_RUN" in proc.stdout

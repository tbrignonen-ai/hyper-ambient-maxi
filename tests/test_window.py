"""Presence view without a display, and a headless window probe."""

import os
import subprocess
import sys
from pathlib import Path

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.ears.hotkey import PynputHotkey, open_hotkey
from hyper_ambient_maxi.memory.store import MemoryStore
from hyper_ambient_maxi.presence import PresenceSession
from hyper_ambient_maxi.presence_window import PresenceView, display_status

ROOT = Path(__file__).resolve().parents[1]


class _Var:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _Factory:
    def __init__(self) -> None:
        self.on_press = None
        self.on_release = None
        self.stopped = False

    def __call__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        return self

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self.stopped = True


class _Key:
    def __init__(self, name: str) -> None:
        self.name = name


def test_view_shows_state_and_the_last_memory_without_tk(tmp_path):
    session = PresenceSession(MemoryStore(tmp_path), ScriptedBrain())
    session.command("/say My dog is called Pixel.")
    labels = {name: _Var() for name in ("state", "reply", "memory")}
    PresenceView(session).paint(labels)
    assert labels["state"].value == "speaking"
    assert "Pixel" in labels["memory"].value
    assert "Pixel" in session.snapshot()
    session.command("/hold")
    PresenceView(session).paint(labels)
    assert labels["state"].value == "listening"
    assert "Pixel" in labels["memory"].value
    assert session.snapshot().splitlines()[0] == "state: listening"


def test_display_and_cli_are_not_run_without_a_display():
    status, reason = display_status()
    headless = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
    if headless:
        assert status == "NOT_RUN"
        assert reason.startswith("NOT_RUN")
    else:
        assert status in {"ok", "NOT_RUN"}
    if not headless:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "hyper_ambient_maxi", "--presence-window", "--home", "/tmp/maxi-window"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert proc.returncode == 3
    assert proc.stdout.startswith("NOT_RUN")


def test_pynput_contract_uses_a_fake_listener_and_stays_not_run_here():
    factory = _Factory()
    hook = PynputHotkey(listener_factory=factory)
    assert hook.poll() is None
    hook.start()
    factory.on_press(_Key("a"))
    factory.on_press(_Key("f9"))
    factory.on_release(_Key("f9"))
    assert hook.poll() == "press"
    assert hook.poll() == "release"
    assert hook.poll() is None
    hook.stop()
    assert factory.stopped is True
    missing = open_hotkey("pynput")
    assert missing.status == "NOT_RUN"
    assert missing.reason.startswith("NOT_RUN")
    assert "pynput" in missing.reason
    assert open_hotkey("auto").status == "NOT_RUN"

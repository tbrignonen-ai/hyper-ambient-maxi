"""Presence states and a hotkey that never grabs the keyboard."""

import os
import subprocess
import sys
from pathlib import Path

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.ears.hotkey import ScriptedHotkey, open_hotkey
from hyper_ambient_maxi.memory.store import MemoryStore
from hyper_ambient_maxi.presence import PresenceSession

ROOT = Path(__file__).resolve().parents[1]


def test_script_stores_a_release_and_a_hold_cancels_speech(tmp_path):
    session = PresenceSession(MemoryStore(tmp_path), ScriptedBrain())
    lines = session.run_script(
        "\n".join(
            [
                "/say My mum was diagnosed with breast cancer last week, and I'm scared.",
                "/say Hello",
                "/hold",
                "/release My dog is called Pixel.",
            ]
        )
    )
    text = "\n".join(lines)
    assert "[speaking]" in text
    assert "breast cancer" in text
    assert "check in" in text
    assert "[barge-in]" in text
    assert "[listening]" in text
    assert session.cuts >= 1
    assert session.last_cut is not None
    played, total, interrupted = session.last_cut
    assert interrupted is True
    assert played < total
    assert session.store.slot("fact:dog-name").value == "Pixel"
    assert session.store.slot("health:mum-diagnosis").value == "breast cancer"


def test_scripted_hotkey_drives_the_same_loop(tmp_path):
    session = PresenceSession(MemoryStore(tmp_path), ScriptedBrain())
    lines = session.drive_hotkey(
        ScriptedHotkey(["press", "release"]),
        {0: "I just got a job at Airbus."},
    )
    assert any("[listening]" in line for line in lines)
    assert session.store.slot("job:employer").value == "Airbus"
    assert open_hotkey("auto").status == "NOT_RUN"
    assert open_hotkey("auto").reason.startswith("NOT_RUN")
    assert open_hotkey("script").status == "ok"


def test_cli_presence_is_not_run_without_a_hook_and_script_works(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    blocked = subprocess.run(
        [sys.executable, "-m", "hyper_ambient_maxi", "--presence", "--home", str(tmp_path / "a")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert blocked.returncode == 3
    assert blocked.stdout.startswith("NOT_RUN")
    scripted = subprocess.run(
        [
            sys.executable,
            "-m",
            "hyper_ambient_maxi",
            "--presence",
            "--hotkey",
            "script",
            "--home",
            str(tmp_path / "b"),
        ],
        input="/say My dog is called Pixel.\n",
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=False,
    )
    assert scripted.returncode == 0, scripted.stderr
    assert "Pixel" in scripted.stdout
    assert "[speaking]" in scripted.stdout
    assert MemoryStore(tmp_path / "b").slot("fact:dog-name").value == "Pixel"

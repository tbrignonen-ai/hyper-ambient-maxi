"""Push-to-talk key as a source of press and release events.

Tests feed ``ScriptedHotkey``. A real keyboard hook is not started in
this process: importing an optional library is not the same as grabbing
F9, and this VM has no reason to. ``open_hotkey`` reports ``NOT_RUN``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HotkeyStatus:
    backend: str
    status: str
    reason: str

    def poll(self) -> str | None:
        return None


class ScriptedHotkey:
    """A list of ``press`` and ``release`` events, consumed in order."""

    backend = "scripted"
    status = "ok"

    def __init__(self, events: list[str]) -> None:
        self._events = list(events)

    def poll(self) -> str | None:
        if not self._events:
            return None
        event = self._events.pop(0)
        if event not in {"press", "release"}:
            raise ValueError(f"unknown hotkey event {event}")
        return event


class PynputHotkey:
    """F9 press and release, through pynput or a test double.

    ``start`` is what installs the hook. Constructing the object and
    ``open_hotkey`` do not. A factory that is not pynput lets tests push
    key events without grabbing the keyboard.
    """

    backend = "pynput"
    status = "ok"

    def __init__(self, key_name: str = "f9", listener_factory=None) -> None:
        self.key_name = key_name.lower()
        self.reason = "pynput hook is not started until start()"
        self._queue: list[str] = []
        self._listener = None
        self._factory = listener_factory
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        factory = self._factory or _pynput_listener
        self._listener = factory(self._on_press, self._on_release)
        start = getattr(self._listener, "start", None)
        if start is not None:
            start()
        self._started = True
        self.reason = f"listening for {self.key_name}"

    def stop(self) -> None:
        if self._listener is not None:
            stop = getattr(self._listener, "stop", None)
            if stop is not None:
                stop()
        self._listener = None
        self._started = False

    def poll(self) -> str | None:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def _on_press(self, key) -> None:
        if _key_name(key) == self.key_name:
            self._queue.append("press")

    def _on_release(self, key) -> None:
        if _key_name(key) == self.key_name:
            self._queue.append("release")


def open_hotkey(backend: str = "auto") -> HotkeyStatus | PynputHotkey:
    """Return a hook only when the caller names one.

    ``auto`` stays ``NOT_RUN`` even if pynput is installed, so a normal
    launch never grabs a key. ``pynput`` reports ``NOT_RUN`` when the
    optional extra is missing and does not call ``start``.
    """
    name = (backend or "auto").strip().lower()
    if name == "script":
        return HotkeyStatus(
            backend="script",
            status="ok",
            reason="reading /hold and /release from the presence script",
        )
    if name == "pynput":
        try:
            import pynput  # noqa: F401
        except Exception as exc:
            return HotkeyStatus(
                backend="pynput",
                status="NOT_RUN",
                reason=(
                    "NOT_RUN: pynput is not installed "
                    f"({exc.__class__.__name__}: {exc}). "
                    "Optional extra: pip install -e '.[hotkey]'. "
                    "This process does not grab a key."
                ),
            )
        hook = PynputHotkey()
        hook.reason = "pynput is installed. start() listens for F9. open_hotkey does not start it."
        return hook
    return HotkeyStatus(
        backend=name or "auto",
        status="NOT_RUN",
        reason=(
            "NOT_RUN: no keyboard hook is installed. Presence tests drive "
            "ScriptedHotkey, and --presence --hotkey script reads /hold "
            "and /release. --hotkey pynput is the optional global key. "
            "This process does not grab a key."
        ),
    )


def _pynput_listener(on_press, on_release):
    from pynput import keyboard

    return keyboard.Listener(on_press=on_press, on_release=on_release)


def _key_name(key) -> str:
    if key is None:
        return ""
    if isinstance(key, str):
        return key.lower()
    name = getattr(key, "name", None)
    if name:
        return str(name).lower()
    char = getattr(key, "char", None)
    if char:
        return str(char).lower()
    return str(key).lower()

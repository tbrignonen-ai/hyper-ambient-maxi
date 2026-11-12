"""Presence window. The state model does not need a display.

``PresenceView`` paints state, the last reply, and the last memory onto
any object with ``set``. Tkinter is optional. With no toolkit or no
display, ``display_status`` is ``NOT_RUN`` and ``open_window`` returns
before ``mainloop``.
"""

from __future__ import annotations

from hyper_ambient_maxi.presence import PresenceSession


class PresenceView:
    def __init__(self, session: PresenceSession) -> None:
        self.session = session

    def fields(self) -> dict[str, str]:
        return {
            "state": self.session.state,
            "reply": self.session.last_reply or "(none)",
            "memory": self.session.last_memory or "(none)",
        }

    def paint(self, labels: dict) -> None:
        for key, value in self.fields().items():
            labels[key].set(value)


def display_status() -> tuple[str, str]:
    try:
        import tkinter as tk
    except Exception as exc:
        return (
            "NOT_RUN",
            "NOT_RUN: tkinter is not installed "
            f"({exc.__class__.__name__}: {exc}). The presence state still updates.",
        )
    try:
        root = tk.Tk()
    except Exception as exc:
        return (
            "NOT_RUN",
            "NOT_RUN: no display for the presence window "
            f"({exc.__class__.__name__}: {exc}).",
        )
    try:
        root.withdraw()
        root.update_idletasks()
        root.destroy()
    except Exception as exc:
        return (
            "NOT_RUN",
            "NOT_RUN: presence window could not open "
            f"({exc.__class__.__name__}: {exc}).",
        )
    return "ok", "tkinter display is available"


def open_window(session: PresenceSession) -> int:
    status, reason = display_status()
    if status != "ok":
        print(reason)
        return 3
    import tkinter as tk

    root = tk.Tk()
    root.title("Hyper Ambient Maxi")
    root.geometry("480x220")
    variables = {name: tk.StringVar() for name in ("state", "reply", "memory")}
    view = PresenceView(session)
    view.paint(variables)
    tk.Label(root, text="Presence", font=("TkDefaultFont", 14)).pack(anchor="w", padx=12, pady=(12, 0))
    tk.Label(root, textvariable=variables["state"]).pack(anchor="w", padx=12)
    tk.Label(root, textvariable=variables["reply"], wraplength=440, justify="left").pack(anchor="w", padx=12, pady=8)
    tk.Label(root, textvariable=variables["memory"], wraplength=440, justify="left").pack(anchor="w", padx=12)

    def refresh() -> None:
        view.paint(variables)
        root.after(200, refresh)

    root.after(200, refresh)
    root.mainloop()
    return 0

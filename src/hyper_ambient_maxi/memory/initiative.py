"""When the harness may speak up. Quiet unless a trigger clears the bar.

Adapted from AI Friend 1.8.3 ``service/initiative.js`` (MIT). The
constraint conflict is the warning the 1.8.3 demo raised when a restart
was about to break a stored rule. Stating the rule ("do not restart…",
"ne … pas") is not itself a conflict. A greeting can also earn one
gentle check-in on the oldest open high-priority thread that is not a
constraint. Constraint threads stay quiet unless the user is about to
act against them.
"""

from __future__ import annotations

import re

from hyper_ambient_maxi.memory.router import is_action, is_greeting
from hyper_ambient_maxi.memory.store import Fact, Thread
from hyper_ambient_maxi.memory.textutil import fold_accents, overlap_score, terms

_PROHIBITION = re.compile(
    r"(?i)^(?:please\s+)?(?:do not|don't|never|ne |n'|il ne faut pas)\b"
)
_GATE = 0.25
_WEIGHT = 4.0
_THRESHOLD = 3.5


def constraint_warning(message: str, facts: list[Fact], *, allow: bool) -> str | None:
    if not allow or not message.strip():
        return None
    if _PROHIBITION.match(message.strip()):
        return None
    if not is_action(message):
        return None
    query = terms(message)
    best: Fact | None = None
    best_relevance = 0.0
    for fact in facts:
        if fact.status != "active" or fact.type != "constraint":
            continue
        relevance = overlap_score(query, f"{fact.summary} {' '.join(fact.tags)}")
        if relevance < _GATE or relevance <= best_relevance:
            continue
        best = fact
        best_relevance = relevance
    if best is None or _WEIGHT < _THRESHOLD:
        return None
    if _french(message):
        return f'Tu es peut-être sur le point d\'aller contre : "{best.summary}".'
    return f'You may be about to act against: "{best.summary}".'


def thread_check_in(message: str, threads: list[Thread], *, allow: bool) -> str | None:
    """One quiet offer. Not a reminder of every open thread."""
    if not allow or not is_greeting(message):
        return None
    query = terms(message)
    candidates = [
        thread
        for thread in threads
        if thread.status == "open"
        and thread.priority == "high"
        and not thread.id.startswith("thread:constraint")
        and overlap_score(query, f"{thread.summary} {thread.title}") < 0.15
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda thread: (thread.created_at, thread.id))
    chosen = candidates[0]
    if _french(message):
        return (
            "Si c'est le moment, prends des nouvelles de ce fil ouvert : "
            f'"{chosen.summary}".'
        )
    return f'If it fits, check in on this open thread: "{chosen.summary}".'


def _french(message: str) -> bool:
    folded = fold_accents(message)
    return bool(
        re.search(
            r"\b(je|mon|ma|mes|tu|ton|ta|chez|passerelle|redemar|salut|bonjour|bonsoir|coucou)\b",
            folded,
        )
    )

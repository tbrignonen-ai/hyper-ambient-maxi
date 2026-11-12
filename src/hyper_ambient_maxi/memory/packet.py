"""Assemble the prompt block. Scores stay out of the text.

Priority, importance, and salience choose the order. They are not
written into the block. A bracket such as ``[high]`` was read aloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from hyper_ambient_maxi.memory.initiative import constraint_warning, thread_check_in
from hyper_ambient_maxi.memory.recall import context_block
from hyper_ambient_maxi.memory.router import route
from hyper_ambient_maxi.memory.salience import select_memories, select_threads
from hyper_ambient_maxi.memory.store import Correction, Fact, MemoryStore, Thread

# Ranking only. The number never appears in the prompt.
_PRIORITY_RANK = {"high": 0, "normal": 1, "medium": 1, "low": 2}


@dataclass
class Packet:
    memories: list[Fact]
    threads: list[Thread]
    warning: str | None
    text: str
    mode: str


def build_packet(
    store: MemoryStore,
    message: str,
    corrections: list[Correction] | None = None,
    embedder=None,
) -> Packet:
    has_constraint = any(fact.type == "constraint" for fact in store.active())
    decision = route(message, has_constraint=has_constraint)
    open_ids = {thread.id for thread in store.open_threads()}
    memories = select_memories(
        list(store.facts.values()),
        message,
        mode=decision.mode,
        limit=decision.max_memories,
        profile=decision.profile_recall,
        open_ids=open_ids,
        embedder=embedder,
    )
    threads = select_threads(
        list(store.threads.values()),
        message,
        mode=decision.mode,
        limit=decision.max_threads,
    )
    allowed = decision.allow_initiative and decision.persist
    warning = constraint_warning(
        message,
        list(store.facts.values()),
        allow=allowed,
    )
    if warning is None:
        warning = thread_check_in(message, store.open_threads(), allow=allowed)
    earlier = prior_values(store, message)
    text = render(memories, threads, warning, corrections or [], earlier=earlier)
    return Packet(memories=memories, threads=threads, warning=warning, text=text, mode=decision.mode)


def render(
    memories: list[Fact],
    threads: list[Thread],
    warning: str | None,
    corrections: list[Correction],
    earlier: list[tuple[str, str]] | None = None,
) -> str:
    # Memories arrive already ordered by salience. Threads are ordered
    # here by priority, and the label is not copied into the line.
    # A thread whose summary already appears as a fact is omitted so the
    # model is not told the same Airbus job twice.
    ordered = sorted(threads, key=lambda thread: (_PRIORITY_RANK.get(thread.priority, 1), thread.id))
    unique = [thread for thread in ordered if not _duplicate_thread(thread, memories)]
    block = context_block(memories, corrections)
    extra: list[str] = []
    if earlier:
        extra.append("Earlier values:")
        for key, value in earlier:
            extra.append(f"- {_earlier_line(key, value)}")
    if unique:
        extra.append("Open threads:")
        for thread in unique:
            extra.append(f"- {thread.summary}")
    if warning:
        extra.append("Worth raising:")
        extra.append(f"- {warning}")
    if not extra:
        return block
    if not block:
        return "<memory>\n" + "\n".join(extra) + "\n</memory>"
    return block.replace("</memory>", "\n" + "\n".join(extra) + "\n</memory>")


_PAST = re.compile(
    r"(?i)\b(?:"
    r"used to|use to|previously|before that|earlier"
    r"|avant|autrefois|précédemment|precedemment"
    r"|ancien(?:ne)?\s+nom|s['’]appelait|l['’]appelais"
    r")\b"
)

_HISTORY_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdog\b|\bpuppy\b|\bpet\b|\bchien\b", re.I), "fact:dog-name"),
    (re.compile(r"\bjob\b|\bwork\b|\bemployer\b|\btravaille\b|\bposte\b|\bemploi\b", re.I), "job:employer"),
    (re.compile(r"\brole\b", re.I), "job:role"),
    (re.compile(r"\bmy name\b|je m['’]appelle|comment je m['’]appelle", re.I), "fact:user-name"),
    (re.compile(r"\bsister\b|\bsœur\b|\bsoeur\b", re.I), "person:sister"),
    (re.compile(r"\bbrother\b|\bfrère\b|\bfrere\b", re.I), "person:brother"),
)


def asks_about_past(message: str) -> bool:
    return _PAST.search(message or "") is not None


def prior_values(store: MemoryStore, message: str) -> list[tuple[str, str]]:
    """Superseded slot values, only when the user asks about the past."""
    if not asks_about_past(message):
        return []
    keys = _history_keys(store, message)
    found: list[tuple[str, str]] = []
    for key in keys:
        for fact in store.history(key):
            if fact.status == "superseded" and str(fact.value).strip():
                found.append((key, fact.value.strip()))
    return found


def prior_notes(store: MemoryStore, message: str) -> list[str]:
    return [value for _key, value in prior_values(store, message)]


def _history_keys(store: MemoryStore, message: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for pattern, key in _HISTORY_HINTS:
        if pattern.search(message or "") and key not in seen:
            seen.add(key)
            keys.append(key)
    if keys:
        return keys
    # A generic "what did I used to..." still deserves earlier values
    # of every slot that has a superseded history.
    for fact in store.facts.values():
        if fact.status == "superseded" and fact.key not in seen:
            seen.add(fact.key)
            keys.append(fact.key)
    return keys


def _earlier_line(key: str, value: str) -> str:
    if key == "fact:dog-name":
        return f"The dog was previously called {value}"
    if key == "fact:user-name":
        return f"The user was previously called {value}"
    if key == "job:employer":
        return f"The user previously worked at {value}"
    if key.startswith("person:"):
        relation = key.split(":", 1)[1].replace("-", " ")
        return f"The user's {relation} was previously called {value}"
    return f"{key} was previously {value}"


def _duplicate_thread(thread: Thread, memories: list[Fact]) -> bool:
    summary = " ".join((thread.summary or "").lower().split())
    if not summary:
        return False
    for fact in memories:
        fact_summary = " ".join((fact.summary or "").lower().split())
        if fact_summary and (summary == fact_summary or summary in fact_summary or fact_summary in summary):
            return True
    return False


"""Assemble the prompt block. Scores stay out of the text.

Priority, importance, and salience choose the order. They are not
written into the block. A bracket such as ``[high]`` was read aloud.
"""

from __future__ import annotations

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
    text = render(memories, threads, warning, corrections or [])
    return Packet(memories=memories, threads=threads, warning=warning, text=text, mode=decision.mode)


def render(
    memories: list[Fact],
    threads: list[Thread],
    warning: str | None,
    corrections: list[Correction],
) -> str:
    # Memories arrive already ordered by salience. Threads are ordered
    # here by priority, and the label is not copied into the line.
    ordered = sorted(threads, key=lambda thread: (_PRIORITY_RANK.get(thread.priority, 1), thread.id))
    block = context_block(memories, corrections)
    extra: list[str] = []
    if ordered:
        extra.append("Open threads:")
        for thread in ordered:
            extra.append(f"- {thread.summary}")
    if warning:
        extra.append("Worth raising:")
        extra.append(f"- {warning}")
    if not extra:
        return block
    if not block:
        return "<memory>\n" + "\n".join(extra) + "\n</memory>"
    return block.replace("</memory>", "\n" + "\n".join(extra) + "\n</memory>")

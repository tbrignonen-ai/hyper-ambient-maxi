"""One turn: extract, commit, verify, then reply.

The model is not asked to remember. That is the failure the 2026-11-05
run hit when a supportive reply was produced and ``friend_remember``
was never called. If the commit cannot be read back, the reply says so
and does not pretend the fact was kept.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyper_ambient_maxi.brain.scripted import ScriptedBrain
from hyper_ambient_maxi.memory.extract import extract
from hyper_ambient_maxi.memory.packet import build_packet, prior_notes
from hyper_ambient_maxi.memory.router import route
from hyper_ambient_maxi.memory.store import (
    Fact,
    MemoryStore,
    MemoryWriteError,
    Receipt,
)

_FAIL = (
    "I heard you, but I could not confirm that was saved. "
    "I am not treating it as remembered."
)


@dataclass
class TurnResult:
    reply: str
    user_text: str
    memory_ok: bool
    brain: str
    voice_status: str
    receipt: Receipt | None
    memories: list[Fact]
    context: str
    error: str | None = None


def run_turn(
    store: MemoryStore,
    brain,
    user_text: str,
    *,
    conversation_id: str = "maxi",
    source_agent: str = "maxi-text",
) -> TurnResult:
    user_text = " ".join((user_text or "").split())
    has_constraint = any(fact.type == "constraint" for fact in store.active())
    decision = route(user_text, has_constraint=has_constraint)
    if not decision.persist:
        result = TurnResult(
            reply="I won't store that.",
            user_text=user_text,
            memory_ok=True,
            brain=getattr(brain, "name", "unknown"),
            voice_status="text",
            receipt=Receipt(stored=0, verified=True),
            memories=[],
            context="",
        )
        _log_turn(store.home, result, conversation_id, source_agent)
        return result
    extracted = extract(user_text)
    try:
        receipt = store.commit(
            extracted.facts,
            source_agent=source_agent,
            conversation_id=conversation_id,
            threads=extracted.threads,
            closes=extracted.closes,
        )
    except MemoryWriteError as exc:
        result = _failed(user_text, str(exc), brain)
        _log_turn(store.home, result, conversation_id, source_agent)
        return result

    if not _obligations_met(store, extracted, receipt):
        result = _failed(
            user_text,
            "durable cue produced no readable fact",
            brain,
        )
        _log_turn(store.home, result, conversation_id, source_agent)
        return result

    packet = build_packet(store, user_text, receipt.corrections, embedder=_embedder())
    memories = packet.memories
    block = packet.text
    notes = prior_notes(store, user_text)
    reply, brain_name = _speak(
        brain,
        user_text,
        block,
        receipt,
        memories,
        notes,
        packet.warning,
        user_name=_address_name(store),
    )
    result = TurnResult(
        reply=reply,
        user_text=user_text,
        memory_ok=True,
        brain=brain_name,
        voice_status="text",
        receipt=receipt,
        memories=memories,
        context=block,
    )
    _log_turn(store.home, result, conversation_id, source_agent)
    return result


def _failed(user_text: str, error: str, brain) -> TurnResult:
    return TurnResult(
        reply=_FAIL,
        user_text=user_text,
        memory_ok=False,
        brain=getattr(brain, "name", "unknown"),
        voice_status="text",
        receipt=None,
        memories=[],
        context="",
        error=error,
    )


def _embedder():
    """Optional local vectors. Unset, or anything other than ``hash``, stays lexical."""
    flag = os.environ.get("HAM_EMBEDDINGS", "").strip().lower()
    if flag != "hash":
        return None
    from hyper_ambient_maxi.memory.embed import HashingEmbedder

    return HashingEmbedder()


def _norm(value: str) -> str:
    return " ".join(value.lower().split())


def _obligations_met(store: MemoryStore, extracted, receipt: Receipt) -> bool:
    """Every parsed fact must be the active slot afterwards.

    ``stored 0`` is acceptable only when the slot was already that value.
    A durable cue the extractor could not structure is returned as an
    episode fact, so it is covered by the same check.
    """
    for prop in extracted.facts:
        active = store.slot(prop.key)
        if active is None or _norm(active.value) != _norm(prop.value):
            return False
    if extracted.facts and receipt.stored == 0 and not receipt.already:
        return False
    return True


def _address_name(store: MemoryStore) -> str:
    """The stored name slot, or empty when no name has been saved."""
    fact = store.slot("fact:user-name")
    if fact is not None and fact.value.strip():
        return fact.value.strip()
    return ""


def _speak(
    brain,
    user,
    context,
    receipt,
    memories,
    notes,
    warning,
    *,
    user_name: str = "",
) -> tuple[str, str]:
    try:
        reply = brain.reply(
            user, context, receipt, memories, notes, warning, user_name=user_name
        )
        return reply.strip(), brain.name
    except Exception:
        if getattr(brain, "name", "") == "scripted":
            raise
        fallback = ScriptedBrain()
        reply = fallback.reply(
            user, context, receipt, memories, notes, warning, user_name=user_name
        )
        return reply.strip(), "scripted-fallback"


def _log_turn(home: Path, result: TurnResult, conversation_id: str, source_agent: str) -> None:
    """One record per turn. AI Friend left orphan turn files whenever
    ``friend_context`` was not paired with ``friend_remember``. Here the
    pair is the function, so the log line exists either way."""
    path = Path(home) / "turns.jsonl"
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "conversation_id": conversation_id,
        "source_agent": source_agent,
        "user": result.user_text,
        "reply": result.reply,
        "memory_ok": result.memory_ok,
        "brain": result.brain,
        "stored_ids": list(result.receipt.ids) if result.receipt else [],
        "error": result.error,
    }
    blob = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)

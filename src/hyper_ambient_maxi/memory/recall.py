"""Pick active memories for a turn, and render them without scores.

The 1.8.3 Linux report (finding 3) showed an 8B model repeating
"relevance 0.58; importance 0.85" because those numbers were in the
prompt. The block this module builds is sentences only.
"""

from __future__ import annotations

import re

from hyper_ambient_maxi.memory.store import Correction, Fact, MemoryStore

_STOP = {
    "i",
    "me",
    "my",
    "you",
    "your",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "what",
    "where",
    "who",
    "when",
    "how",
    "about",
    "remember",
    "please",
    "that",
    "it",
    "this",
    "to",
    "of",
    "in",
    "on",
    "and",
    "or",
    "for",
    "with",
    "have",
    "has",
}

_HINTS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bdog\b|\bpuppy\b|\bpet\b", re.I), ("fact:dog-name",)),
    (re.compile(r"\bmum\b|\bmom\b|\bmother\b", re.I), ("health:mum-diagnosis",)),
    (re.compile(r"\bdad\b|\bfather\b", re.I), ("health:dad-diagnosis",)),
    (re.compile(r"\bjob\b|\bwork\b|\bemployer\b", re.I), ("job:",)),
    (re.compile(r"\brole\b", re.I), ("job:role",)),
    (re.compile(r"\bcodename\b", re.I), ("fact:test-codename",)),
    (re.compile(r"\bprefer\b|\bcoffee\b|\bmilk\b", re.I), ("preference:",)),
)
_PROFILE = re.compile(r"what do you remember|what do you know about me", re.I)


def recall(store: MemoryStore, query: str, limit: int = 8) -> list[Fact]:
    """Salience-ranked active facts. The cap comes from the router."""
    from hyper_ambient_maxi.memory.packet import build_packet

    return build_packet(store, query).memories[:limit]


def context_block(
    facts: list[Fact],
    corrections: list[Correction] | None = None,
) -> str:
    corrections = corrections or []
    if not facts and not corrections:
        return ""
    lines = ["<memory>", "What you know about the user:"]
    for fact in facts:
        lines.append(f"- {fact.summary}")
    for correction in corrections:
        lines.append(
            f"- Corrected {correction.key}: now {correction.new_value} "
            f"(was {correction.old_value})."
        )
    lines.append("</memory>")
    return "\n".join(lines)


def _hints(query: str) -> list[str]:
    found: list[str] = []
    for pattern, keys in _HINTS:
        if pattern.search(query):
            found.extend(keys)
    return found


def _key_match(key: str, hints: list[str]) -> bool:
    for hint in hints:
        if hint.endswith(":"):
            if key.startswith(hint):
                return True
        elif key == hint:
            return True
    return False


def _newest(facts: list[Fact]) -> list[Fact]:
    return sorted(facts, key=lambda fact: fact.created_at, reverse=True)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9']+", text)
        if token.lower() not in _STOP and len(token) > 2
    }


def _overlap(query: str, fact: Fact) -> bool:
    query_tokens = _tokens(query)
    fact_tokens = _tokens(f"{fact.summary} {fact.value} {fact.key}")
    return bool(query_tokens & fact_tokens)

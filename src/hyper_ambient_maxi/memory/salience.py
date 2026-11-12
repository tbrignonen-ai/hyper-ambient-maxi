"""Which memories and threads fill a small packet.

Adapted from AI Friend 1.8.3 ``service/salience.js`` (MIT). The score is
the published v1 formula. The model-facing text is not: 1.8.3 printed
``relevance 0.58; importance 0.85`` and an 8B model read it aloud.
Ranking stays here. The renderer in ``packet.py`` drops the numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from hyper_ambient_maxi.memory.store import Fact, Thread
from hyper_ambient_maxi.memory.textutil import overlap_score, terms

_DAY_S = 86400.0
_THRESHOLDS = {"friend": 0.03, "light": 0.08}
_PRIORITY = {"high": 1.0, "normal": 0.6, "medium": 0.6, "low": 0.3}
_PROFILE_TYPES = {"personal_fact", "health", "job", "preference", "constraint"}


@dataclass
class ScoredMemory:
    fact: Fact
    score: float


@dataclass
class ScoredThread:
    thread: Thread
    score: float


# Slot keys are English. French questions use the other word, and the
# summary stays in the language of the write. These aliases are scoring
# only; they are not copied into the prompt.
_SLOT_ALIASES = {
    "fact:dog-name": "dog chien chienne puppy pet",
    "health:mum-diagnosis": "mum mom mother mere maman diagnosis sante",
    "health:dad-diagnosis": "dad father pere papa diagnosis sante",
    "job:employer": "job work employer travail travaille poste emploi employeur",
    "job:role": "role poste metier",
    "fact:user-name": "name nom appelle appellent",
    "fact:home-city": "home city ville habite live",
    "health:allergy": "allergy allergic allergique",
    "fact:appointment": "appointment rendez-vous medecin",
}


def score_memory(
    fact: Fact,
    query: set[str],
    open_ids: set[str],
    now: datetime,
    *,
    message: str = "",
    embedder=None,
) -> float | None:
    alias = _SLOT_ALIASES.get(fact.key, "")
    if fact.key.startswith("preference:"):
        alias = "prefer preference prefere aime"
    haystack = f"{fact.summary} {fact.value} {' '.join(fact.tags)} {alias}"
    relevance = overlap_score(query, haystack)
    if relevance <= 0 and embedder is not None and message.strip():
        relevance = _embedded_relevance(embedder, message, haystack)
    if relevance <= 0:
        return None
    importance = _importance(fact.importance)
    confidence = _unit(fact.confidence, 0.8)
    age = _recency(fact.created_at, now)
    linked = bool(fact.thread_id and fact.thread_id in open_ids)
    commitment = fact.type in {"commitment", "promise", "deadline"}
    return (
        relevance
        * (0.45 + 0.55 * importance)
        * (0.70 + 0.30 * confidence)
        * age
        * (1.2 if linked else 1.0)
        * (1.25 if commitment else 1.0)
    )


def score_thread(thread: Thread, query: set[str], now: datetime) -> float:
    priority = _PRIORITY.get(thread.priority, 0.6)
    relevance = overlap_score(query, f"{thread.summary} {thread.title}")
    age = _recency(thread.created_at, now)
    return (0.55 + 0.45 * priority) * (0.6 + 0.4 * relevance) * age


def select_memories(
    facts: list[Fact],
    message: str,
    *,
    mode: str,
    limit: int,
    profile: bool,
    open_ids: set[str],
    now: datetime | None = None,
    embedder=None,
) -> list[Fact]:
    if mode == "passthrough" or limit <= 0:
        return []
    now = now or datetime.now(timezone.utc)
    query = terms(message)
    threshold = _THRESHOLDS.get(mode, 0.08)
    scored: list[ScoredMemory] = []
    for fact in facts:
        if fact.status != "active":
            continue
        value = score_memory(
            fact, query, open_ids, now, message=message, embedder=embedder
        )
        if value is None or value < threshold:
            continue
        scored.append(ScoredMemory(fact, value))
    scored.sort(key=lambda item: (-item.score, item.fact.id))
    chosen = [item.fact for item in scored[:limit]]
    if profile and len(chosen) < limit:
        seen = {fact.id for fact in chosen}
        rest = [
            fact
            for fact in facts
            if fact.status == "active" and fact.type in _PROFILE_TYPES and fact.id not in seen
        ]
        rest.sort(key=lambda fact: (-_importance(fact.importance), -_unit(fact.confidence, 0.8), fact.id))
        chosen.extend(rest[: limit - len(chosen)])
    return chosen


def select_threads(
    threads: list[Thread],
    message: str,
    *,
    mode: str,
    limit: int,
    now: datetime | None = None,
) -> list[Thread]:
    if mode == "passthrough" or limit <= 0:
        return []
    now = now or datetime.now(timezone.utc)
    query = terms(message)
    scored: list[ScoredThread] = []
    for thread in threads:
        if thread.status != "open":
            continue
        item = ScoredThread(thread, score_thread(thread, query, now))
        if mode != "friend" and overlap_score(query, f"{thread.summary} {thread.title}") <= 0:
            continue
        scored.append(item)
    scored.sort(key=lambda item: (-item.score, item.thread.id))
    return [item.thread for item in scored[:limit]]


def _embedded_relevance(embedder, message: str, haystack: str) -> float:
    """Cosine only when lexical overlap is zero. A failure stays lexical."""
    try:
        from hyper_ambient_maxi.memory.embed import cosine

        left = embedder.embed(message)
        right = embedder.embed(haystack)
        score = cosine(left, right)
    except Exception:
        return 0.0
    if score < 0.55:
        return 0.0
    return score


def _importance(value: float) -> float:
    if value > 1:
        value = value / 5
    return min(1.0, max(0.0, value))


def _unit(value: float, fallback: float) -> float:
    if value is None or isinstance(value, float) and math.isnan(value):
        value = fallback
    return min(1.0, max(0.0, value))


def _recency(created_at: str, now: datetime) -> float:
    try:
        stamp = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.7
    age_days = max(0.0, (now - stamp).total_seconds() / _DAY_S)
    return 0.6 + 0.4 * math.exp(-age_days / 90)

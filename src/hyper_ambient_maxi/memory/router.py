"""How much of the memory layer a turn receives.

Adapted from AI Friend 1.8.3 ``service/router.js`` (MIT). The rule is the
same: continuity >= 3 is ``friend``, context >= 1 is ``light``, otherwise
``passthrough``. French cues sit next to the English ones. A stored
constraint plus an action verb promotes the turn to ``friend`` so the
warning can fire without the user saying "remember".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hyper_ambient_maxi.memory.textutil import fold_accents

_MAX = 4000

_PROFILE = re.compile(
    r"(?:"
    r"\b(?:what|anything|everything) (?:do |did |else do |can )?you (?:remember|know|recall) (?:about|of) (?:me|myself)\b"
    r"|^\s*(?:so,? )?what (?:do|did|else do|can) you (?:remember|know|recall)\s*[?.!]*\s*$"
    r"|\btell me (?:what|everything|anything) you (?:remember|know|recall)(?: about (?:me|myself))?\s*[?.!]*\s*$"
    r"|\bdo you (?:remember|know) (?:anything )?about me\b"
    r"|qu'est-ce que tu (?:te rappelles|sais|te souviens)"
    r"|que sais-tu (?:de|sur) moi"
    r"|tu te souviens de quoi"
    r")",
    re.I,
)
_PRIVACY = re.compile(
    r"(?i)\b(?:forget this|don't remember this|do not remember this|off the record|no memory|oublie ça|oublie ca|n'enregistre pas|hors dossier)\b"
)
_DONT_REMIND = re.compile(r"(?i)\b(?:don't remind me|do not remind me|ne me rappelle pas)\b")
_PERSONAL = re.compile(
    r"(?i)\b(?:my (?:dog|mum|mom|mother|dad)|i prefer|i work|diagnos|"
    r"mon chien|ma mère|ma mere|ma maman|mon père|mon pere|je préfère|je prefere|"
    r"je travaille|diagnostiqu|j'ai un chien|mon nom de code)\b"
)
_PREFERENCE = re.compile(r"(?i)\b(?:i prefer|i like|je préfère|je prefere|j'aime)\b")
_UTILITY = re.compile(r"(?i)\b(?:translate|summarise|summarize|convert|traduis|résume|resume)\b")
_ACTION = re.compile(
    r"(?i)\b(?:restart|deploy|delete|drop|wipe|reset|redemar\w*|supprim\w*|effac\w*|arret\w*)\b"
)


@dataclass
class Decision:
    mode: str
    continuity: float
    context: float
    max_memories: int
    max_threads: int
    allow_initiative: bool
    persist: bool
    confidence: float
    source: str = "router"
    reasons: list[str] = field(default_factory=list)
    profile_recall: bool = False


def is_profile_recall(message: str) -> bool:
    return _PROFILE.search(fold_accents(message)[:_MAX]) is not None or _PROFILE.search(message[:_MAX]) is not None


_GREETING = re.compile(
    r"(?i)^\s*(?:hi|hello|hey|good morning|good evening|"
    r"salut|bonjour|bonsoir|coucou|ça va|ca va|tu vas bien)\s*[!.?]*\s*$"
)


def is_action(message: str) -> bool:
    return _ACTION.search(fold_accents(message)) is not None


def is_greeting(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return _GREETING.match(text) is not None or _GREETING.match(fold_accents(text)) is not None


def route(message: str, *, has_constraint: bool = False) -> Decision:
    text = (message or "")[:_MAX]
    folded = fold_accents(text)
    if _PRIVACY.search(text) or _PRIVACY.search(folded):
        return Decision(
            mode="passthrough",
            continuity=0,
            context=0,
            max_memories=0,
            max_threads=0,
            allow_initiative=False,
            persist=False,
            confidence=0.99,
            source="privacy",
            reasons=["privacy"],
        )
    continuity = 0.0
    context = 0.0
    reasons: list[str] = []
    profile = is_profile_recall(text)
    if profile:
        continuity += 3
        reasons.append("profile_recall")
    if _PERSONAL.search(text) or _PERSONAL.search(folded):
        continuity += 3
        reasons.append("personal_fact")
    if re.search(r"(?i)\b(?:changed my mind|no longer|not anymore|j'ai changé|je ne .+ plus)\b", text):
        continuity += 3
        reasons.append("contradiction")
    if re.search(r"(?i)^(?:please\s+)?(?:do not|don't|never|ne |n'|il ne faut pas)\b", text.strip()):
        continuity += 3
        reasons.append("constraint_statement")
    if has_constraint and is_action(text):
        continuity = max(continuity, 3)
        reasons.append("constraint_watch")
    if _PREFERENCE.search(text) or _PREFERENCE.search(folded):
        context += 2
        reasons.append("preference")
    if re.search(r"(?i)\b(?:need to|have to|must|je dois|il faut)\b", text):
        context += 1
        reasons.append("obligation")
    questioning = "?" in text or bool(
        re.match(r"(?i)(?:what|where|who|when|why|how|comment|où|ou|pourquoi|que )\b", text.strip())
    )
    greeting = is_greeting(text)
    if greeting:
        reasons.append("greeting")
    if continuity >= 3:
        mode = "friend"
    elif (context >= 1 or questioning or greeting) and not _UTILITY.search(folded):
        mode = "light"
        if questioning and context < 1 and not greeting:
            reasons.append("question")
    else:
        mode = "passthrough"
    caps = {
        "friend": (3, 5, True),
        "light": (2, 2, False),
        "passthrough": (0, 0, False),
    }[mode]
    allow = (caps[2] or greeting) and _DONT_REMIND.search(text) is None and _DONT_REMIND.search(folded) is None
    return Decision(
        mode=mode,
        continuity=continuity,
        context=context,
        max_memories=caps[0],
        max_threads=caps[1],
        allow_initiative=allow,
        persist=True,
        confidence=0.6 if mode == "passthrough" else 0.9,
        reasons=reasons,
        profile_recall=profile,
    )

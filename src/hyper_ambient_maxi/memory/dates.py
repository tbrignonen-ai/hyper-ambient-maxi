"""Relative dates in French appointment sentences.

``demain`` is the next calendar day. ``jeudi prochain`` is the next
Thursday after today, and if today is already Thursday it is one week
on. The clock is an argument so a test can pin the day. Nothing here
calls a network time service.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from hyper_ambient_maxi.memory.textutil import fold_accents

_WEEKDAYS = {
    "lundi": 0,
    "mardi": 1,
    "mercredi": 2,
    "jeudi": 3,
    "vendredi": 4,
    "samedi": 5,
    "dimanche": 6,
}
_NEXT = re.compile(
    r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+prochain\b"
)


def resolve_fr_when(text: str, today: date | None = None) -> date | None:
    today = today or date.today()
    folded = fold_accents(text or "")
    if re.search(r"\bdemain\b", folded):
        return today + timedelta(days=1)
    if re.search(r"\baujourd'?hui\b", folded):
        return today
    match = _NEXT.search(folded)
    if match is None:
        return None
    target = _WEEKDAYS[match.group(1)]
    delta = (target - today.weekday()) % 7
    if delta == 0:
        delta = 7
    return today + timedelta(days=delta)

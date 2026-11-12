"""Lexical overlap used by salience and the constraint check.

Adapted from AI Friend 1.8.3 ``service/text.js`` (MIT, Copyright 2026
AI Friend contributors). The English stemmer and the edit-distance
fallback are that algorithm. Accent folding is added so a French
inflection still meets the same overlap gate.
"""

from __future__ import annotations

import math
import re

STOP = {
    "about", "after", "again", "also", "and", "are", "been", "but", "can",
    "could", "did", "does", "for", "from", "have", "how", "into", "its",
    "just", "more", "not", "our", "that", "the", "their", "then", "there",
    "these", "this", "those", "was", "were", "what", "when", "where",
    "which", "with", "would", "you", "your",
    # French function words. Content words (chien, mère, travail) stay.
    "les", "des", "une", "mon", "mes", "ton", "tes", "son", "ses", "elle",
    "nous", "vous", "pas", "plus", "dans", "pour", "avec", "chez", "est",
    "suis", "que", "qui", "quoi", "dont", "cette", "ceci", "cela", "aux",
    "sur", "par", "une", "moi", "toi",
}
_PLURAL_EXCEPTIONS = {"series", "species", "news"}
_SUFFIXES = (("ment", 4, 7), ("ness", 4, 7), ("ing", 3, 6), ("ed", 2, 5), ("ly", 2, 5))
_FUZZY_THRESHOLD = 0.72
_FUZZY_WEIGHT = 0.75
_FUZZY_MIN = 6
_ACCENTS = str.maketrans(
    {
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "à": "a", "â": "a", "ä": "a",
        "î": "i", "ï": "i",
        "ô": "o", "ö": "o",
        "ù": "u", "û": "u", "ü": "u",
        "ç": "c", "ÿ": "y",
    }
)


def fold_accents(text: str) -> str:
    lowered = text.lower().replace("œ", "oe").replace("æ", "ae")
    return lowered.translate(_ACCENTS)


def terms(text: str) -> set[str]:
    source = fold_accents(text)
    words = {
        word
        for word in re.findall(r"[a-z0-9_-]{3,}", source)
        if word not in STOP
    }
    words.update(re.findall(r"\b\d{1,4}\b", source))
    folded = set()
    for word in words:
        stem = _fold(word)
        if stem and len(stem) >= 3 and stem not in STOP:
            folded.add(stem)
    words.update(folded)
    return words


def overlap_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    words = terms(text)
    if not words:
        return 0.0
    overlap = 0.0
    for word in query_terms:
        if word in words:
            overlap += 1.0
            continue
        best = 0.0
        for candidate in words:
            best = max(best, _similar(word, candidate))
        if best >= _FUZZY_THRESHOLD:
            overlap += _FUZZY_WEIGHT * best
    return min(1.0, overlap / math.sqrt(len(query_terms) * len(words)))


def _fold(word: str) -> str | None:
    if len(word) < 4 or word in _PLURAL_EXCEPTIONS:
        return None
    for suffix, strip, minimum in _SUFFIXES:
        if len(word) >= minimum and word.endswith(suffix):
            stem = word[: -strip]
            return stem if len(stem) >= 3 else None
    if re.search(r"[^aeiou]ies$", word):
        return word[:-3] + "y"
    if re.search(r"(?:ch|sh|ss|x|z|s)es$", word):
        return word[:-2]
    if re.search(r"(?:ss|us|is)$", word):
        return None
    if word.endswith("s"):
        return word[:-1]
    return None


def _similar(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if len(left) >= 5 and len(right) >= 5 and (left.startswith(right) or right.startswith(left)):
        return 0.9
    if len(left) < _FUZZY_MIN or len(right) < _FUZZY_MIN:
        return 0.0
    return 1.0 - _edit_distance(left, right) / max(len(left), len(right))


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    rows = len(left) + 1
    cols = len(right) + 1
    grid = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        grid[i][0] = i
    for j in range(cols):
        grid[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            grid[i][j] = min(
                grid[i - 1][j] + 1,
                grid[i][j - 1] + 1,
                grid[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and left[i - 1] == right[j - 2]
                and left[i - 2] == right[j - 1]
            ):
                grid[i][j] = min(grid[i][j], grid[i - 2][j - 2] + 1)
    return grid[-1][-1]

"""Optional local vectors. Tests never open a network connection.

The default ranker stays lexical. ``HAM_EMBEDDINGS=hash`` blends in a
deterministic hashing embedder that lives in this process. A caller can
also pass its own embedder. If ``embed`` raises or the cosine is weak,
salience keeps the lexical score, which is zero when the words do not
overlap, so a broken embedder cannot invent a match the words refused.
"""

from __future__ import annotations

import math
import zlib

from hyper_ambient_maxi.memory.textutil import terms


class HashingEmbedder:
    """Bag of signed feature hashes. Same text, same vector, no weights."""

    def __init__(self, dim: int = 64) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for term in terms(text):
            slot = zlib.crc32(term.encode("utf-8")) % self.dim
            sign = 1.0 if zlib.crc32(b"+" + term.encode("utf-8")) % 2 == 0 else -1.0
            vector[slot] += sign
        return _normalize(vector)


def cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    score = dot / math.sqrt(left_norm * right_norm)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]

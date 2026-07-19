"""Reciprocal Rank Fusion — слияние ранжированных списков (гибридный поиск).

score(id) = Σ по спискам 1/(K + rank). Классическая константа K=60 сглаживает
вклад хвоста; списки могут быть разной длины, id встречаться не во всех.
"""

from __future__ import annotations

from collections import defaultdict

RRF_K = 60


def rrf_merge(rankings: list[list[str]], limit: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1.0 / (RRF_K + rank + 1)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [item_id for item_id, _ in ordered[:limit]]

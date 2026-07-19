"""Общие текстовые утилиты: FTS5-запрос из свободного текста.

Префиксы слов — грубая, но эффективная замена стемминга для русской
морфологии («Ивана» → «иван*» находит Иван/Иваном/Ивану). Используется
памятью и лексической половиной гибридного поиска RAG.
"""

from __future__ import annotations

import re

STOPWORDS = {
    "что", "как", "это", "где", "когда", "кто", "про", "обо", "или", "еще",
    "ещё", "знаешь", "помнишь", "расскажи", "мне", "меня", "тебя", "есть",
    "было", "были", "такое", "такой", "такая", "ты", "вы", "the", "about",
}


def build_fts_query(text: str) -> str | None:
    """Текст запроса → FTS5-выражение из префиксов значимых слов."""
    words = re.findall(r"\w+", text.lower())
    significant = [w for w in words if len(w) >= 3 and w not in STOPWORDS]
    if not significant:
        return None
    parts = []
    for word in significant[:8]:
        # срезаем 1–2 буквы окончания, но оставляем минимум 4 (у коротких — всё слово)
        prefix = word[: max(4, len(word) - 2)] if len(word) >= 5 else word[:3]
        parts.append(f'"{prefix}"*')
    return " OR ".join(parts)

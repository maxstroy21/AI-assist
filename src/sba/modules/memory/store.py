"""Хранилище памяти: SQLite + FTS5.

Факты не удаляются физически: «забудь» ставит retracted_at, вытеснение —
superseded_by (история решений сохраняется, docs/02-architecture.md §3).
Поиск — FTS5 с префиксами слов: грубая, но эффективная замена стемминга
для русской морфологии («Ивана» → «иван*» находит Иван/Иваном/Ивану).
Векторный recall придёт в Sprint 4 поверх этого же хранилища.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sba.infra.db import Database

STOPWORDS = {
    "что", "как", "это", "где", "когда", "кто", "про", "обо", "или", "еще",
    "ещё", "знаешь", "помнишь", "расскажи", "мне", "меня", "тебя", "есть",
    "было", "были", "такое", "такой", "такая", "ты", "вы", "the", "about",
}

ACTIVE = "superseded_by IS NULL AND retracted_at IS NULL"


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


@dataclass(frozen=True)
class FactView:
    id: str
    type: str
    subject: str
    content: str
    created_at: str


def _view(row: object) -> FactView:
    return FactView(
        id=row["id"],  # type: ignore[index]
        type=row["type"],  # type: ignore[index]
        subject=row["subject"],  # type: ignore[index]
        content=row["content"],  # type: ignore[index]
        created_at=row["created_at"],  # type: ignore[index]
    )


class MemoryStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        user_id: str,
        fact_type: str,
        subject: str,
        content: str,
        source: str = "explicit",
        confidence: float = 1.0,
    ) -> FactView:
        fact_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        if fact_type == "preference":
            # предпочтение по той же теме вытесняет старое (актуально одно)
            await self._db.execute(
                f"UPDATE memory_facts SET superseded_by=?, updated_at=?"
                f" WHERE user_id=? AND type='preference'"
                f" AND lower(subject)=lower(?) AND {ACTIVE}",
                (fact_id, now, user_id, subject),
            )
        await self._db.execute(
            "INSERT INTO memory_facts"
            " (id, user_id, type, subject, content, source, confidence,"
            "  created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, user_id, fact_type, subject, content, source, confidence, now, now),
        )
        await self._db.execute(
            "INSERT INTO memory_fts (subject, content, fact_id) VALUES (?, ?, ?)",
            (subject, content, fact_id),
        )
        return FactView(
            id=fact_id, type=fact_type, subject=subject, content=content, created_at=now
        )

    async def search(self, user_id: str, query: str, k: int = 6) -> list[FactView]:
        fts = build_fts_query(query)
        if fts is None:
            # «что ты помнишь?» — показываем свежие факты
            rows = await self._db.fetch_all(
                f"SELECT * FROM memory_facts WHERE user_id=? AND {ACTIVE}"
                f" ORDER BY created_at DESC LIMIT ?",
                (user_id, k),
            )
            return [_view(r) for r in rows]
        rows = await self._db.fetch_all(
            f"SELECT f.* FROM memory_fts m JOIN memory_facts f ON f.id = m.fact_id"
            f" WHERE memory_fts MATCH ? AND f.user_id=? AND {ACTIVE}"
            f" ORDER BY rank LIMIT ?",
            (fts, user_id, k),
        )
        return [_view(r) for r in rows]

    async def retract(self, user_id: str, query: str) -> list[FactView]:
        """Мягкое «забудь»: помечает найденные факты retracted_at."""
        matched = await self.search(user_id, query, k=20)
        if build_fts_query(query) is None:
            return []  # «забудь всё» без уточнения — не выполняем
        now = datetime.now(UTC).isoformat()
        for fact in matched:
            await self._db.execute(
                "UPDATE memory_facts SET retracted_at=?, updated_at=? WHERE id=?",
                (now, now, fact.id),
            )
        return matched

    async def preferences(self, user_id: str) -> list[FactView]:
        rows = await self._db.fetch_all(
            f"SELECT * FROM memory_facts"
            f" WHERE user_id=? AND type='preference' AND {ACTIVE}"
            f" ORDER BY created_at",
            (user_id,),
        )
        return [_view(r) for r in rows]

    async def count_active(self, user_id: str) -> int:
        row = await self._db.fetch_one(
            f"SELECT COUNT(*) AS n FROM memory_facts WHERE user_id=? AND {ACTIVE}",
            (user_id,),
        )
        return int(row["n"]) if row else 0

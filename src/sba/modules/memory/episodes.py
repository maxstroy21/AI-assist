"""Эпизодическая память: суммаризации прошлых разговоров (Sprint 7).

Эпизод — итог одного разговора: о чём говорили, темы, проект. Пишется
консолидацией (consolidation.py), ищется FTS-запросом при вопросах вида
«о чём мы говорили…». Один разговор → не больше одного эпизода
(UNIQUE conversation_id — идемпотентность повторной консолидации).

Отступление от 02-architecture §3: векторная копия эпизода в Qdrant
отложена — первичный recall идёт по фактам (они гибридные), эпизоды
ищутся лексически; на машине владельца semantic и так выключен.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sba.infra.db import Database
from sba.infra.text import build_fts_query


@dataclass(frozen=True)
class EpisodeView:
    id: str
    conversation_id: str
    summary: str
    topics: str          # темы через запятую (как хранится)
    project: str | None
    closed_at: str       # ISO конца разговора — «когда это было»


def _view(row: object) -> EpisodeView:
    return EpisodeView(
        id=row["id"],  # type: ignore[index]
        conversation_id=row["conversation_id"],  # type: ignore[index]
        summary=row["summary"],  # type: ignore[index]
        topics=row["topics"],  # type: ignore[index]
        project=row["project"],  # type: ignore[index]
        closed_at=row["closed_at"],  # type: ignore[index]
    )


class EpisodeStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        user_id: str,
        conversation_id: str,
        summary: str,
        topics: list[str],
        project: str | None,
        started_at: str,
        closed_at: str,
    ) -> EpisodeView | None:
        """Сохранить эпизод; None — эпизод этого разговора уже есть."""
        existing = await self._db.fetch_one(
            "SELECT id FROM episodes WHERE conversation_id=?", (conversation_id,)
        )
        if existing is not None:
            return None
        episode_id = uuid.uuid4().hex
        topics_text = ", ".join(topics)
        await self._db.execute(
            "INSERT INTO episodes (id, user_id, conversation_id, summary, topics,"
            " project, started_at, closed_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                episode_id, user_id, conversation_id, summary, topics_text,
                project, started_at, closed_at, datetime.now(UTC).isoformat(),
            ),
        )
        await self._db.execute(
            "INSERT INTO episodes_fts (summary, topics, episode_id) VALUES (?, ?, ?)",
            (summary, topics_text, episode_id),
        )
        return EpisodeView(
            id=episode_id, conversation_id=conversation_id, summary=summary,
            topics=topics_text, project=project, closed_at=closed_at,
        )

    async def search(self, user_id: str, query: str, k: int = 3) -> list[EpisodeView]:
        fts = build_fts_query(query)
        if fts is None:
            return []
        rows = await self._db.fetch_all(
            "SELECT e.* FROM episodes_fts f JOIN episodes e ON e.id = f.episode_id"
            " WHERE episodes_fts MATCH ? AND e.user_id=?"
            " ORDER BY rank LIMIT ?",
            (fts, user_id, k),
        )
        return [_view(r) for r in rows]

    async def recent(self, user_id: str, days: int = 7, limit: int = 10) -> list[EpisodeView]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        rows = await self._db.fetch_all(
            "SELECT * FROM episodes WHERE user_id=? AND created_at>=?"
            " ORDER BY closed_at DESC LIMIT ?",
            (user_id, cutoff, limit),
        )
        return [_view(r) for r in rows]

    async def count(self, user_id: str) -> int:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) AS n FROM episodes WHERE user_id=?", (user_id,)
        )
        return int(row["n"]) if row else 0

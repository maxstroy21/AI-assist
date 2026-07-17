"""Чтение истории диалога для построения контекста.

Таблица messages принадлежит ядру (пишет Router, читает Context Builder).
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

from sba.infra.db import Database


class HistoryEntry(NamedTuple):
    role: str  # user | assistant
    content: str


class HistoryReader(Protocol):
    async def recent(self, conversation_id: str, limit: int) -> list[HistoryEntry]: ...


class HistoryStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def recent(self, conversation_id: str, limit: int) -> list[HistoryEntry]:
        """Последние сообщения разговора в хронологическом порядке."""
        rows = await self._db.fetch_all(
            "SELECT role, content FROM messages WHERE conversation_id=?"
            " ORDER BY rowid DESC LIMIT ?",
            (conversation_id, limit),
        )
        return [HistoryEntry(role=r["role"], content=r["content"]) for r in reversed(rows)]

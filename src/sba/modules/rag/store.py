"""Хранилище чанков модуля rag: SQLite `rag_chunks` + FTS5 (лексическая
половина гибридного поиска). Таблицы принадлежат этому модулю (правило границ №2).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sba.infra.db import Database
from sba.infra.text import build_fts_query
from sba.modules.rag.interface import Chunk


@dataclass(frozen=True)
class StoredChunk:
    id: str  # канонический UUID — он же id точки в Qdrant
    file_id: str
    seq: int
    text: str
    locator: str
    path: str


def _stored(row: object) -> StoredChunk:
    return StoredChunk(
        id=row["id"],  # type: ignore[index]
        file_id=row["file_id"],  # type: ignore[index]
        seq=row["seq"],  # type: ignore[index]
        text=row["text"],  # type: ignore[index]
        locator=row["locator"],  # type: ignore[index]
        path=row["path"],  # type: ignore[index]
    )


_SELECT = (
    "SELECT c.id, c.file_id, c.seq, c.text, c.locator, f.path"
    " FROM rag_chunks c JOIN rag_files f ON f.id = c.file_id"
)


class ChunkStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def replace_for_file(self, file_id: str, chunks: list[Chunk]) -> list[StoredChunk]:
        """Атомарно заменяет чанки файла; возвращает записи с новыми id."""
        old_ids = await self.delete_for_file(file_id)
        del old_ids  # id точек Qdrant удаляет вызывающий (RAGService) ДО замены
        stored: list[StoredChunk] = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            await self._db.execute(
                "INSERT INTO rag_chunks (id, file_id, seq, text, locator)"
                " VALUES (?, ?, ?, ?, ?)",
                (chunk_id, file_id, chunk.seq, chunk.text, chunk.locator),
            )
            await self._db.execute(
                "INSERT INTO rag_chunks_fts (text, chunk_id) VALUES (?, ?)",
                (chunk.text, chunk_id),
            )
            stored.append(
                StoredChunk(
                    id=chunk_id,
                    file_id=file_id,
                    seq=chunk.seq,
                    text=chunk.text,
                    locator=chunk.locator,
                    path="",
                )
            )
        return stored

    async def ids_for_file(self, file_id: str) -> list[str]:
        rows = await self._db.fetch_all(
            "SELECT id FROM rag_chunks WHERE file_id=?", (file_id,)
        )
        return [str(r["id"]) for r in rows]

    async def delete_for_file(self, file_id: str) -> list[str]:
        ids = await self.ids_for_file(file_id)
        if ids:
            await self._db.execute(
                "DELETE FROM rag_chunks_fts WHERE chunk_id IN"
                " (SELECT id FROM rag_chunks WHERE file_id=?)",
                (file_id,),
            )
            await self._db.execute("DELETE FROM rag_chunks WHERE file_id=?", (file_id,))
        return ids

    async def lexical_search(self, query: str, limit: int) -> list[StoredChunk]:
        fts = build_fts_query(query)
        if fts is None:
            return []
        rows = await self._db.fetch_all(
            f"{_SELECT} JOIN rag_chunks_fts m ON c.id = m.chunk_id"
            f" WHERE rag_chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts, limit),
        )
        return [_stored(r) for r in rows]

    async def by_ids(self, ids: list[str]) -> dict[str, StoredChunk]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = await self._db.fetch_all(
            f"{_SELECT} WHERE c.id IN ({placeholders})", tuple(ids)
        )
        return {chunk.id: chunk for chunk in (_stored(r) for r in rows)}

    async def count(self) -> int:
        row = await self._db.fetch_one("SELECT COUNT(*) AS n FROM rag_chunks")
        return int(row["n"]) if row else 0

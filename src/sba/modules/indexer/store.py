"""Каталог файлов и очередь индексации (SQLite, переживают рестарт).

Каталог — источник истины о том, что проиндексировано (path, hash, mtime,
статус); очередь дедуплицируется по пути: файл либо ждёт индексации, либо нет.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sba.infra.db import Database


@dataclass(frozen=True)
class FileRecord:
    id: str
    path: str
    content_hash: str | None
    mtime: float | None
    size: int | None
    status: str  # pending | indexed | failed
    error: str | None
    chunk_count: int


@dataclass(frozen=True)
class QueueItem:
    path: str
    op: str  # upsert | delete
    attempts: int


def _record(row: object) -> FileRecord:
    return FileRecord(
        id=row["id"],  # type: ignore[index]
        path=row["path"],  # type: ignore[index]
        content_hash=row["content_hash"],  # type: ignore[index]
        mtime=row["mtime"],  # type: ignore[index]
        size=row["size"],  # type: ignore[index]
        status=row["status"],  # type: ignore[index]
        error=row["error"],  # type: ignore[index]
        chunk_count=row["chunk_count"],  # type: ignore[index]
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


class CatalogStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ── каталог ──────────────────────────────────────────────────────────────

    async def get(self, path: str) -> FileRecord | None:
        row = await self._db.fetch_one("SELECT * FROM rag_files WHERE path=?", (path,))
        return _record(row) if row else None

    async def all_files(self) -> list[FileRecord]:
        rows = await self._db.fetch_all("SELECT * FROM rag_files ORDER BY path")
        return [_record(r) for r in rows]

    async def upsert_pending(
        self, path: str, content_hash: str, mtime: float, size: int
    ) -> FileRecord:
        """Регистрирует новую/изменённую версию файла со статусом pending."""
        existing = await self.get(path)
        if existing is None:
            file_id = uuid.uuid4().hex
            await self._db.execute(
                "INSERT INTO rag_files"
                " (id, path, content_hash, mtime, size, status, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (file_id, path, content_hash, mtime, size, _now()),
            )
        else:
            file_id = existing.id
            await self._db.execute(
                "UPDATE rag_files SET content_hash=?, mtime=?, size=?,"
                " status='pending', error=NULL, updated_at=? WHERE id=?",
                (content_hash, mtime, size, _now(), file_id),
            )
        record = await self.get(path)
        assert record is not None
        return record

    async def touch(self, path: str, mtime: float, size: int) -> None:
        """mtime изменился, содержимое нет — просто обновляем метаданные."""
        await self._db.execute(
            "UPDATE rag_files SET mtime=?, size=?, updated_at=? WHERE path=?",
            (mtime, size, _now(), path),
        )

    async def mark_indexed(self, path: str, chunk_count: int) -> None:
        now = _now()
        await self._db.execute(
            "UPDATE rag_files SET status='indexed', error=NULL, chunk_count=?,"
            " indexed_at=?, updated_at=? WHERE path=?",
            (chunk_count, now, now, path),
        )

    async def mark_failed(self, path: str, error: str) -> None:
        await self._db.execute(
            "UPDATE rag_files SET status='failed', error=?, updated_at=? WHERE path=?",
            (error[:500], _now(), path),
        )

    async def remove(self, path: str) -> None:
        await self._db.execute("DELETE FROM rag_files WHERE path=?", (path,))

    async def status_counts(self) -> dict[str, int]:
        rows = await self._db.fetch_all(
            "SELECT status, COUNT(*) AS n FROM rag_files GROUP BY status"
        )
        return {str(r["status"]): int(r["n"]) for r in rows}

    # ── очередь ──────────────────────────────────────────────────────────────

    async def enqueue(self, path: str, op: str) -> None:
        await self._db.execute(
            "INSERT INTO rag_queue (path, op, enqueued_at, attempts)"
            " VALUES (?, ?, ?, 0)"
            " ON CONFLICT(path) DO UPDATE SET op=excluded.op,"
            "   enqueued_at=excluded.enqueued_at",
            (path, op, _now()),
        )

    async def queue_batch(self, limit: int) -> list[QueueItem]:
        rows = await self._db.fetch_all(
            "SELECT path, op, attempts FROM rag_queue ORDER BY enqueued_at LIMIT ?",
            (limit,),
        )
        return [
            QueueItem(path=str(r["path"]), op=str(r["op"]), attempts=int(r["attempts"]))
            for r in rows
        ]

    async def queue_remove(self, path: str) -> None:
        await self._db.execute("DELETE FROM rag_queue WHERE path=?", (path,))

    async def queue_bump_attempts(self, path: str) -> None:
        await self._db.execute(
            "UPDATE rag_queue SET attempts = attempts + 1 WHERE path=?", (path,)
        )

    async def queue_size(self) -> int:
        row = await self._db.fetch_one("SELECT COUNT(*) AS n FROM rag_queue")
        return int(row["n"]) if row else 0

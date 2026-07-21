"""Undo-журнал файловых операций: SQLite, батчи + записи.

Батч — одна операция глазами владельца («переместить файл», «архивировать
12 файлов»); записи — конкретные пары src → dst. Журналируются и сухие
прогоны (dry_run=1) — планы видны в /fileops, но откату не подлежат:
откатывать нечего, файлы не менялись.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sba.infra.db import Database


@dataclass(frozen=True)
class BatchView:
    id: str
    kind: str            # move | copy | rename | archive | undo
    dry_run: bool
    description: str
    undo_of: str | None
    created_at: str
    undone_at: str | None


@dataclass(frozen=True)
class EntryView:
    batch_id: str
    seq: int
    op: str              # move | copy
    src: str
    dst: str
    executed: bool
    error: str | None


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _batch(row: object) -> BatchView:
    return BatchView(
        id=row["id"],  # type: ignore[index]
        kind=row["kind"],  # type: ignore[index]
        dry_run=bool(row["dry_run"]),  # type: ignore[index]
        description=row["description"],  # type: ignore[index]
        undo_of=row["undo_of"],  # type: ignore[index]
        created_at=row["created_at"],  # type: ignore[index]
        undone_at=row["undone_at"],  # type: ignore[index]
    )


def _entry(row: object) -> EntryView:
    return EntryView(
        batch_id=row["batch_id"],  # type: ignore[index]
        seq=row["seq"],  # type: ignore[index]
        op=row["op"],  # type: ignore[index]
        src=row["src"],  # type: ignore[index]
        dst=row["dst"],  # type: ignore[index]
        executed=bool(row["executed"]),  # type: ignore[index]
        error=row["error"],  # type: ignore[index]
    )


class FileOpsStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def create_batch(
        self,
        kind: str,
        dry_run: bool,
        description: str,
        entries: list[tuple[str, str, str]],  # (op, src, dst)
        undo_of: str | None = None,
    ) -> str:
        batch_id = uuid.uuid4().hex
        await self._db.execute(
            "INSERT INTO file_ops_batches (id, kind, dry_run, description, undo_of,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (batch_id, kind, int(dry_run), description, undo_of, _utcnow()),
        )
        for seq, (op, src, dst) in enumerate(entries):
            await self._db.execute(
                "INSERT INTO file_ops_entries (batch_id, seq, op, src, dst)"
                " VALUES (?, ?, ?, ?, ?)",
                (batch_id, seq, op, src, dst),
            )
        return batch_id

    async def mark_entry(self, batch_id: str, seq: int, error: str | None = None) -> None:
        """Итог записи: без error — выполнена; с error — не выполнена, причина."""
        await self._db.execute(
            "UPDATE file_ops_entries SET executed=?, error=? WHERE batch_id=? AND seq=?",
            (int(error is None), error, batch_id, seq),
        )

    async def entries(self, batch_id: str) -> list[EntryView]:
        rows = await self._db.fetch_all(
            "SELECT * FROM file_ops_entries WHERE batch_id=? ORDER BY seq", (batch_id,)
        )
        return [_entry(r) for r in rows]

    async def last_undoable(self) -> BatchView | None:
        """Последний выполненный (не план, не сам undo) и ещё не отменённый батч."""
        row = await self._db.fetch_one(
            "SELECT * FROM file_ops_batches WHERE dry_run=0 AND undone_at IS NULL"
            " AND kind != 'undo' ORDER BY rowid DESC LIMIT 1"
        )
        return _batch(row) if row else None

    async def mark_undone(self, batch_id: str) -> None:
        await self._db.execute(
            "UPDATE file_ops_batches SET undone_at=? WHERE id=?", (_utcnow(), batch_id)
        )

    async def recent(self, limit: int = 8) -> list[BatchView]:
        rows = await self._db.fetch_all(
            "SELECT * FROM file_ops_batches ORDER BY rowid DESC LIMIT ?", (limit,)
        )
        return [_batch(r) for r in rows]

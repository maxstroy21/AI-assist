"""Хранилище джобов планировщика: SQLite, переживает рестарт.

Момент срабатывания хранится в UTC (ISO) — сравнения корректны при любом
часовом поясе владельца; RRULE считается в его поясе (dtstart хранит пояс).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sba.infra.db import Database


@dataclass(frozen=True)
class JobView:
    id: str
    topic: str
    payload: dict[str, str]
    next_fire_at: datetime          # aware, UTC
    rrule: str | None
    dtstart: datetime | None        # aware, пояс владельца — якорь RRULE
    misfire: str                    # deliver | skip


def _view(row: object) -> JobView:
    raw_dtstart = row["dtstart"]  # type: ignore[index]
    return JobView(
        id=row["id"],  # type: ignore[index]
        topic=row["topic"],  # type: ignore[index]
        payload=json.loads(row["payload"]),  # type: ignore[index]
        next_fire_at=datetime.fromisoformat(row["next_fire_at"]),  # type: ignore[index]
        rrule=row["rrule"],  # type: ignore[index]
        dtstart=datetime.fromisoformat(raw_dtstart) if raw_dtstart else None,
        misfire=row["misfire"],  # type: ignore[index]
    )


class SchedulerStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def upsert(
        self,
        job_id: str | None,
        topic: str,
        payload: dict[str, str],
        next_fire_at: datetime,
        rrule: str | None = None,
        dtstart: datetime | None = None,
        misfire: str = "deliver",
    ) -> str:
        job_id = job_id or uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            "INSERT INTO scheduler_jobs"
            " (id, topic, payload, next_fire_at, rrule, dtstart, misfire, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET topic=excluded.topic, payload=excluded.payload,"
            "  next_fire_at=excluded.next_fire_at, rrule=excluded.rrule,"
            "  dtstart=excluded.dtstart, misfire=excluded.misfire, updated_at=excluded.updated_at",
            (
                job_id,
                topic,
                json.dumps(payload, ensure_ascii=False),
                next_fire_at.astimezone(UTC).isoformat(),
                rrule,
                dtstart.isoformat() if dtstart else None,
                misfire,
                now,
                now,
            ),
        )
        return job_id

    async def due(self, now: datetime) -> list[JobView]:
        rows = await self._db.fetch_all(
            "SELECT * FROM scheduler_jobs WHERE next_fire_at <= ? ORDER BY next_fire_at",
            (now.astimezone(UTC).isoformat(),),
        )
        return [_view(r) for r in rows]

    async def get(self, job_id: str) -> JobView | None:
        row = await self._db.fetch_one("SELECT * FROM scheduler_jobs WHERE id=?", (job_id,))
        return _view(row) if row else None

    async def delete(self, job_id: str) -> None:
        await self._db.execute("DELETE FROM scheduler_jobs WHERE id=?", (job_id,))

    # Условные операции: «пока джоб не перепланировали из обработчика».
    # Обработчик события может пересоздать джоб с тем же id (follow-up);
    # безусловное удаление после публикации снесло бы его новую версию.

    async def delete_if_unchanged(self, job_id: str, next_fire_at: datetime) -> None:
        await self._db.execute(
            "DELETE FROM scheduler_jobs WHERE id=? AND next_fire_at=?",
            (job_id, next_fire_at.astimezone(UTC).isoformat()),
        )

    async def advance_if_unchanged(
        self, job_id: str, old_fire_at: datetime, new_fire_at: datetime
    ) -> None:
        await self._db.execute(
            "UPDATE scheduler_jobs SET next_fire_at=?, updated_at=?"
            " WHERE id=? AND next_fire_at=?",
            (
                new_fire_at.astimezone(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
                job_id,
                old_fire_at.astimezone(UTC).isoformat(),
            ),
        )

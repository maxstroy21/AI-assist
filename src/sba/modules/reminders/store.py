"""Хранилище напоминаний и журнала доставки.

reminder_log — ключ идемпотентности (риск Sprint 6 из роадмапа): доставка
фиксируется по (reminder_id, scheduled_for); повторное срабатывание после
сбоя видит запись и не шлёт дубль. Статусы напоминания:
scheduled — ждёт срабатывания; fired — доставлено, ждёт реакции;
done / cancelled — финал (✅ / ✖), физически не удаляется.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sba.infra.db import Database

SHORT_ID_LEN = 6  # как у задач: столько первых hex-символов показываем
_SCAN_LIMIT = 500


@dataclass(frozen=True)
class ReminderView:
    id: str
    user_id: str
    text: str
    task_id: str | None
    due: str                   # ISO с часовым поясом владельца
    rrule: str | None
    followup_minutes: int | None
    status: str
    job_id: str | None
    source_message_id: str | None
    created_at: str

    @property
    def short_id(self) -> str:
        return self.id[:SHORT_ID_LEN]

    @property
    def active(self) -> bool:
        return self.status in ("scheduled", "fired")


def _view(row: object) -> ReminderView:
    return ReminderView(
        id=row["id"],  # type: ignore[index]
        user_id=row["user_id"],  # type: ignore[index]
        text=row["text"],  # type: ignore[index]
        task_id=row["task_id"],  # type: ignore[index]
        due=row["due"],  # type: ignore[index]
        rrule=row["rrule"],  # type: ignore[index]
        followup_minutes=row["followup_minutes"],  # type: ignore[index]
        status=row["status"],  # type: ignore[index]
        job_id=row["job_id"],  # type: ignore[index]
        source_message_id=row["source_message_id"],  # type: ignore[index]
        created_at=row["created_at"],  # type: ignore[index]
    )


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class ReminderStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        user_id: str,
        text: str,
        due: str,
        task_id: str | None = None,
        rrule: str | None = None,
        followup_minutes: int | None = None,
        job_id: str | None = None,
        source_message_id: str | None = None,
    ) -> ReminderView:
        reminder_id = uuid.uuid4().hex
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO reminders (id, user_id, text, task_id, due, rrule,"
            " followup_minutes, status, job_id, source_message_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', ?, ?, ?, ?)",
            (reminder_id, user_id, text, task_id, due, rrule,
             followup_minutes, job_id, source_message_id, now, now),
        )
        found = await self.get(reminder_id)
        assert found is not None
        return found

    async def get(self, reminder_id: str) -> ReminderView | None:
        row = await self._db.fetch_one(
            "SELECT * FROM reminders WHERE id=?", (reminder_id,)
        )
        return _view(row) if row else None

    async def update(
        self,
        reminder_id: str,
        *,
        text: str | None = None,
        due: str | None = None,
        rrule: str | None = None,
        clear_rrule: bool = False,
        status: str | None = None,
        job_id: str | None = None,
    ) -> ReminderView | None:
        fields: dict[str, object] = {}
        if text is not None:
            fields["text"] = text
        if due is not None:
            fields["due"] = due
        if rrule is not None or clear_rrule:
            fields["rrule"] = rrule
        if status is not None:
            fields["status"] = status
        if job_id is not None:
            fields["job_id"] = job_id
        if not fields:
            return await self.get(reminder_id)
        fields["updated_at"] = _utcnow()
        assignments = ", ".join(f"{name}=?" for name in fields)
        await self._db.execute(
            f"UPDATE reminders SET {assignments} WHERE id=?",
            (*fields.values(), reminder_id),
        )
        return await self.get(reminder_id)

    async def find_by_task(self, task_id: str) -> ReminderView | None:
        """Действующее авто-напоминание задачи (scheduled — уникально по индексу)."""
        row = await self._db.fetch_one(
            "SELECT * FROM reminders WHERE task_id=? AND status='scheduled'", (task_id,)
        )
        return _view(row) if row else None

    async def list_active(self, user_id: str, limit: int = 15) -> list[ReminderView]:
        """Предстоящие (scheduled) по сроку; fired не показываем — они уже пришли."""
        rows = await self._db.fetch_all(
            "SELECT * FROM reminders WHERE user_id=? AND status='scheduled'"
            f" ORDER BY due LIMIT {_SCAN_LIMIT}",
            (user_id,),
        )
        return [_view(r) for r in rows][:limit]

    async def resolve(self, user_id: str, ref: str, limit: int = 5) -> list[ReminderView]:
        """Напоминание по ссылке модели: короткий id или слова из текста.
        Ищем среди активных (scheduled | fired) — с ними есть что делать."""
        ref = ref.strip().lstrip("#")
        if ref and all(c in "0123456789abcdef" for c in ref.lower()) and len(ref) >= 4:
            rows = await self._db.fetch_all(
                "SELECT * FROM reminders WHERE user_id=? AND id LIKE ?",
                (user_id, f"{ref.lower()}%"),
            )
            if rows:
                return [_view(r) for r in rows]
        rows = await self._db.fetch_all(
            "SELECT * FROM reminders WHERE user_id=? AND status IN ('scheduled','fired')"
            f" ORDER BY due LIMIT {_SCAN_LIMIT}",
            (user_id,),
        )
        needle = " ".join(ref.lower().split())
        matched = [r for r in (_view(row) for row in rows) if needle in r.text.lower()]
        return matched[:limit]

    async def count_active(self, user_id: str) -> int:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) AS n FROM reminders WHERE user_id=? AND status='scheduled'",
            (user_id,),
        )
        return int(row["n"]) if row else 0

    # ── журнал доставки (идемпотентность) ────────────────────────────────────

    async def try_log_delivery(
        self, reminder_id: str, scheduled_for: str, detail: str
    ) -> bool:
        """Записать доставку; False — этот момент уже обработан (дубль после сбоя)."""
        before = await self._db.fetch_one(
            "SELECT 1 AS x FROM reminder_log WHERE reminder_id=? AND scheduled_for=?",
            (reminder_id, scheduled_for),
        )
        if before is not None:
            return False
        await self._db.execute(
            "INSERT OR IGNORE INTO reminder_log"
            " (reminder_id, scheduled_for, delivered_at, detail) VALUES (?, ?, ?, ?)",
            (reminder_id, scheduled_for, _utcnow(), detail),
        )
        return True

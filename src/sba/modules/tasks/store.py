"""Хранилище задач: SQLite + FTS5 (префиксный поиск под русскую морфологию).

Задачи не удаляются физически: закрытие и отмена — смена статуса
(open | done | cancelled), история остаётся для аудита и статистики.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sba.infra.db import Database
from sba.infra.text import build_fts_query

SHORT_ID_LEN = 6  # столько первых hex-символов id показываем пользователю
_SCAN_LIMIT = 500  # потолок выборки до Python-фильтров (личный масштаб задач)


@dataclass(frozen=True)
class TaskView:
    id: str
    title: str
    notes: str
    project: str | None
    status: str
    due: str | None            # ISO с часовым поясом владельца
    rrule: str | None
    source_message_id: str | None
    created_at: str
    completed_at: str | None

    @property
    def short_id(self) -> str:
        return self.id[:SHORT_ID_LEN]


def _view(row: object) -> TaskView:
    return TaskView(
        id=row["id"],  # type: ignore[index]
        title=row["title"],  # type: ignore[index]
        notes=row["notes"],  # type: ignore[index]
        project=row["project"],  # type: ignore[index]
        status=row["status"],  # type: ignore[index]
        due=row["due"],  # type: ignore[index]
        rrule=row["rrule"],  # type: ignore[index]
        source_message_id=row["source_message_id"],  # type: ignore[index]
        created_at=row["created_at"],  # type: ignore[index]
        completed_at=row["completed_at"],  # type: ignore[index]
    )


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class TaskStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        user_id: str,
        title: str,
        notes: str = "",
        project: str | None = None,
        due: str | None = None,
        rrule: str | None = None,
        source_message_id: str | None = None,
    ) -> TaskView:
        task_id = uuid.uuid4().hex
        now = _utcnow()
        await self._db.execute(
            "INSERT INTO tasks (id, user_id, title, notes, project, status, due, rrule,"
            " source_message_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)",
            (task_id, user_id, title, notes, project, due, rrule, source_message_id, now, now),
        )
        await self._db.execute(
            "INSERT INTO tasks_fts (title, notes, project, task_id) VALUES (?, ?, ?, ?)",
            (title, notes, project or "", task_id),
        )
        found = await self.get(user_id, task_id)
        assert found is not None
        return found

    async def get(self, user_id: str, task_id: str) -> TaskView | None:
        row = await self._db.fetch_one(
            "SELECT * FROM tasks WHERE user_id=? AND id=?", (user_id, task_id)
        )
        return _view(row) if row else None

    async def update(
        self,
        user_id: str,
        task_id: str,
        *,
        title: str | None = None,
        notes: str | None = None,
        project: str | None = None,
        status: str | None = None,
        due: str | None = None,
        clear_due: bool = False,
        rrule: str | None = None,
        completed_at: str | None = None,
    ) -> TaskView | None:
        """Точечное обновление; None = «не менять» (снять срок — clear_due)."""
        fields: dict[str, object] = {}
        if title is not None:
            fields["title"] = title
        if notes is not None:
            fields["notes"] = notes
        if project is not None:
            fields["project"] = project or None
        if status is not None:
            fields["status"] = status
        if due is not None or clear_due:
            fields["due"] = due
            fields["rrule"] = rrule  # срок и повторение меняются вместе
        if completed_at is not None:
            fields["completed_at"] = completed_at
        if not fields:
            return await self.get(user_id, task_id)
        fields["updated_at"] = _utcnow()
        assignments = ", ".join(f"{name}=?" for name in fields)
        await self._db.execute(
            f"UPDATE tasks SET {assignments} WHERE user_id=? AND id=?",
            (*fields.values(), user_id, task_id),
        )
        task = await self.get(user_id, task_id)
        if task is not None and ("title" in fields or "notes" in fields or "project" in fields):
            await self._db.execute("DELETE FROM tasks_fts WHERE task_id=?", (task_id,))
            await self._db.execute(
                "INSERT INTO tasks_fts (title, notes, project, task_id) VALUES (?, ?, ?, ?)",
                (task.title, task.notes, task.project or "", task_id),
            )
        return task

    async def search(
        self,
        user_id: str,
        query: str = "",
        project: str | None = None,
        status: str | None = "open",
        limit: int = 15,
    ) -> list[TaskView]:
        """Поиск: FTS по тексту + фильтры; без текста — список по сроку."""
        where = ["t.user_id=?"]
        params: list[object] = [user_id]
        if status is not None:
            where.append("t.status=?")
            params.append(status)
        fts = build_fts_query(query) if query else None
        if fts is None:
            rows = await self._db.fetch_all(
                f"SELECT t.* FROM tasks t WHERE {' AND '.join(where)}"
                f" ORDER BY t.due IS NULL, t.due, t.created_at LIMIT {_SCAN_LIMIT}",
                tuple(params),
            )
        else:
            rows = await self._db.fetch_all(
                f"SELECT t.* FROM tasks_fts f JOIN tasks t ON t.id = f.task_id"
                f" WHERE tasks_fts MATCH ? AND {' AND '.join(where)}"
                f" ORDER BY rank LIMIT {_SCAN_LIMIT}",
                (fts, *params),
            )
        tasks = [_view(r) for r in rows]
        if project:
            # фильтр в Python: lower() SQLite не понимает кириллицу,
            # а «экспедиция» должна находить проект «Экспедиция-2026»
            needle = project.lower()
            tasks = [t for t in tasks if needle in (t.project or "").lower()]
        return tasks[:limit]

    async def resolve(self, user_id: str, ref: str, limit: int = 5) -> list[TaskView]:
        """Найти открытую задачу по ссылке модели: короткий id или слова названия."""
        ref = ref.strip().lstrip("#")
        if ref and all(c in "0123456789abcdef" for c in ref.lower()) and len(ref) >= 4:
            rows = await self._db.fetch_all(
                "SELECT * FROM tasks WHERE user_id=? AND id LIKE ?",
                (user_id, f"{ref.lower()}%"),
            )
            if rows:
                return [_view(r) for r in rows]
        found = await self.search(user_id, query=ref, status="open", limit=limit)
        if found:
            return found
        # FTS не дружит с очень короткими словами — пробуем подстроку названия
        # (фильтр в Python: lower() SQLite не понимает кириллицу)
        rows = await self._db.fetch_all(
            f"SELECT * FROM tasks WHERE user_id=? AND status='open'"
            f" ORDER BY created_at LIMIT {_SCAN_LIMIT}",
            (user_id,),
        )
        needle = ref.lower()
        matched = [t for t in (_view(r) for r in rows) if needle in t.title.lower()]
        return matched[:limit]

    async def count_open(self, user_id: str) -> int:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) AS n FROM tasks WHERE user_id=? AND status='open'", (user_id,)
        )
        return int(row["n"]) if row else 0

"""Событие изменения задачи (Sprint 6): Reminder Engine поддерживает
авто-напоминания к срокам задач, не будучи вызванным напрямую
(слабая связанность через шину, docs/02-architecture.md §2.2).
"""

from __future__ import annotations

from typing import Literal

from sba.core.events import Event


class TaskChanged(Event):
    reason: Literal["created", "updated", "completed"]
    task_id: str
    title: str
    status: str            # open | done | cancelled
    due: str | None        # ISO с часовым поясом владельца
    rrule: str | None

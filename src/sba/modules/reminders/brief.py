"""Утренняя сводка (Sprint 6): задачи на сегодня, просроченные, напоминания дня.

Сводка детерминированная, без LLM — отступление от «профиля morning-brief»
(см. docs/06-roadmap.md): по принципу П-1 (docs/01-requirements.md) регулярная
доставка не должна зависеть от модели, а слабая 7B-модель могла бы приукрасить
список дел. Misfire-политика джоба — skip: сводка, пропущенная больше чем на
grace (ноутбук был выключен), вечером уже не нужна.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, time, tzinfo

import structlog

from sba.core.types import OutgoingKind, OutgoingMessage
from sba.infra.audit import AuditLog
from sba.modules.reminders.store import ReminderStore
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.tasks.service import TasksService

log = structlog.get_logger(__name__)

BRIEF_TOPIC = "brief.morning"
BRIEF_JOB_ID = "morning_brief"
_LOG_ID = "brief:morning"   # ключ идемпотентности в reminder_log
_MAX_LINES = 8              # сводка должна оставаться короткой

MONTHS_GEN = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
WEEKDAYS_NOM = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
)


def parse_brief_time(value: str) -> time:
    """«08:30» → time; валидация выполнена конфигом, здесь только разбор."""
    hour, minute = value.split(":", 1)
    return time(int(hour), int(minute))


class MorningBrief:
    def __init__(
        self,
        scheduler: SchedulerService,
        store: ReminderStore,
        timezone: tzinfo,
        deliver: Callable[[OutgoingMessage], Awaitable[None]],
        targets: list[tuple[str, str]],
        audit: AuditLog,
        at: time,
        tasks: TasksService | None = None,
        owner_id: str = "owner",
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._tz = timezone
        self._deliver = deliver
        self._targets = targets
        self._audit = audit
        self._at = at
        self._tasks = tasks
        self._owner = owner_id

    async def schedule(self) -> None:
        """Идемпотентная регистрация ежедневного джоба (вызывается на старте).

        Существующий джоб с тем же временем не переписывается: иначе рестарт
        после пропущенного утра сдвинул бы срабатывание на завтра и misfire-
        политика (доставить в пределах grace) не получила бы шанса."""
        existing = await self._scheduler.get(BRIEF_JOB_ID)
        if (
            existing is not None
            and existing.rrule == "FREQ=DAILY"
            and existing.dtstart is not None
            and (existing.dtstart.hour, existing.dtstart.minute)
            == (self._at.hour, self._at.minute)
        ):
            return
        dtstart = datetime.now(self._tz).replace(
            hour=self._at.hour, minute=self._at.minute, second=0, microsecond=0
        )
        await self._scheduler.schedule_rrule(
            BRIEF_TOPIC, {}, rrule="FREQ=DAILY", dtstart=dtstart,
            job_id=BRIEF_JOB_ID, misfire="skip",
        )

    async def on_job_fired(self, event: JobFired) -> None:
        if event.topic != BRIEF_TOPIC:
            return
        scheduled_iso = event.scheduled_for.isoformat()
        if not await self._store.try_log_delivery(_LOG_ID, scheduled_iso, "delivered"):
            log.info("brief_duplicate_suppressed")
            return
        text = await self.build_text()
        for channel, user_id in self._targets:
            await self._deliver(
                OutgoingMessage(
                    user_id=user_id,
                    channel=channel,
                    text=text,
                    kind=OutgoingKind.NOTIFICATION,
                )
            )
        await self._audit.record("morning_brief_sent", "reminders", text[:200])

    async def build_text(self) -> str:
        now = datetime.now(self._tz)
        header = (
            f"🌅 Доброе утро! Сегодня {now.day} {MONTHS_GEN[now.month - 1]}, "
            f"{WEEKDAYS_NOM[now.weekday()]}."
        )
        sections: list[str] = []

        if self._tasks is not None:
            open_tasks = await self._tasks.search(status="open", limit=200)
            today: list[str] = []
            overdue = 0
            for task in open_tasks:
                if not task.due:
                    continue
                due = datetime.fromisoformat(task.due).astimezone(self._tz)
                if due.date() == now.date():
                    today.append(
                        f"• {due.hour:02d}:{due.minute:02d} — {task.title}"
                    )
                elif due < now:
                    overdue += 1
            if today:
                sections.append(
                    "Задачи на сегодня:\n" + "\n".join(today[:_MAX_LINES])
                )
            if overdue:
                sections.append(
                    f"⚠️ Просроченных задач: {overdue} — список: /tasks"
                )

        upcoming = await self._store.list_active(self._owner, limit=50)
        today_reminders = []
        for reminder in upcoming:
            due = datetime.fromisoformat(reminder.due).astimezone(self._tz)
            if due.date() == now.date() and reminder.task_id is None:
                # напоминания к задачам уже видны в разделе задач — не дублируем
                today_reminders.append(
                    f"• {due.hour:02d}:{due.minute:02d} — {reminder.text}"
                )
        if today_reminders:
            sections.append(
                "Напоминания сегодня:\n" + "\n".join(today_reminders[:_MAX_LINES])
            )

        if not sections:
            sections.append("Задач со сроком и напоминаний на сегодня нет. Свободный день!")
        return "\n\n".join([header, *sections])

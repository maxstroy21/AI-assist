"""Reminder Engine: ассистент сам приходит к владельцу вовремя (Sprint 6).

Устройство:
- напоминание — сущность в SQLite; срабатывание — джоб в Scheduler
  (топик reminder.fire, полезная нагрузка — id напоминания);
- доставка — OutgoingMessage(kind=NOTIFICATION) через Router во все
  каналы доставки, с кнопками ✅ Сделал / ⏰ Позже / ✖ Отменить;
- идемпотентность — reminder_log: повторное срабатывание одного и того же
  момента после сбоя не шлёт дубль (риск Sprint 6 из роадмапа);
- follow-up «если не сделал — напомни снова»: после доставки напоминание
  переназначается через followup_minutes, пока владелец не нажмёт ✅/✖
  (у связанного с задачей — пока задача не закрыта);
- авто-напоминания: задача со сроком получает напоминание на срок
  (по событию TaskChanged и стартовой синхронизацией);
- «умный момент» без срока: ближайшее утро (default_hour) — детерминированная
  эвристика вместо LLM (недоверие к модели + бюджет CPU).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

import structlog

from sba.core.types import MessageAction, OutgoingKind, OutgoingMessage
from sba.infra.audit import AuditLog
from sba.llm.gateway import LLMError
from sba.modules.reminders.store import ReminderStore, ReminderView
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.tasks.dates import (
    WhenParseError,
    WhenParser,
    describe_rrule,
    next_occurrence,
)
from sba.modules.tasks.events import TaskChanged
from sba.modules.tasks.service import WEEKDAY_SHORT, TasksService

log = structlog.get_logger(__name__)

OWNER_ID = "owner"  # один владелец (допущение A-3), как в задачах и памяти

REMINDER_TOPIC = "reminder.fire"
LATE_NOTICE_AFTER = timedelta(minutes=15)  # с какого опоздания честно помечаем доставку

ACTION_DONE = "done"
ACTION_SNOOZE = "snooze"
ACTION_CANCEL = "cancel"


@dataclass(frozen=True)
class ReminderOutcome:
    reminder: ReminderView | None
    question: str | None = None   # переспрос (срок неоднозначен) — не создано
    error: str | None = None      # разбор срока недоступен — не создано


class ReminderService:
    def __init__(
        self,
        store: ReminderStore,
        scheduler: SchedulerService,
        parser: WhenParser,
        timezone: tzinfo,
        deliver: Callable[[OutgoingMessage], Awaitable[None]],
        targets: list[tuple[str, str]],
        audit: AuditLog,
        tasks: TasksService | None = None,
        snooze_minutes: int = 180,
        followup_minutes: int = 24 * 60,
        default_hour: int = 9,
        list_limit: int = 15,
        owner_id: str = OWNER_ID,
    ) -> None:
        self._store = store
        self._scheduler = scheduler
        self._parser = parser
        self._tz = timezone
        self._deliver = deliver
        self._targets = targets      # (канал, user_id) — каналы доставки уведомлений
        self._audit = audit
        self._tasks = tasks
        self._snooze_minutes = snooze_minutes
        self._followup_minutes = followup_minutes
        self._default_hour = default_hour
        self._list_limit = list_limit
        self._owner = owner_id

    # ── создание и изменение (инструменты) ───────────────────────────────────

    async def create(
        self,
        text: str,
        when: str = "",
        repeat_until_done: bool = False,
        source_message_id: str | None = None,
    ) -> ReminderOutcome:
        try:
            parsed = await self._parser.parse(when)
        except (WhenParseError, LLMError) as exc:
            log.warning("reminder_when_failed", when=when, error=str(exc))
            return ReminderOutcome(
                reminder=None,
                error=f"не смог разобрать срок «{when}» — повторите точнее, "
                "например «завтра в 9» или «через 2 часа»",
            )
        if parsed.kind == "unclear":
            return ReminderOutcome(reminder=None, question=parsed.question)
        if parsed.kind == "none" or parsed.due is None:
            due = self._smart_moment()
            rrule = None
        else:
            due = parsed.due
            rrule = parsed.rrule
        reminder = await self._store.add(
            self._owner,
            text=text,
            due=due.isoformat(),
            rrule=rrule,
            followup_minutes=self._followup_minutes if repeat_until_done else None,
            source_message_id=source_message_id,
        )
        await self._schedule(reminder)
        return ReminderOutcome(reminder=reminder)

    def _smart_moment(self) -> datetime:
        """«Умный момент» без срока: ближайшее утро в default_hour."""
        now = datetime.now(self._tz)
        candidate = now.replace(
            hour=self._default_hour, minute=0, second=0, microsecond=0
        )
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    async def _schedule(self, reminder: ReminderView) -> None:
        due = datetime.fromisoformat(reminder.due)
        job_id = f"rem:{reminder.id}"
        if reminder.rrule:
            await self._scheduler.schedule_rrule(
                REMINDER_TOPIC, {"reminder_id": reminder.id},
                rrule=reminder.rrule, dtstart=due, job_id=job_id,
            )
        else:
            await self._scheduler.schedule_once(
                REMINDER_TOPIC, {"reminder_id": reminder.id}, at=due, job_id=job_id
            )
        await self._store.update(reminder.id, job_id=job_id)

    async def _cancel_jobs(self, reminder: ReminderView) -> None:
        await self._scheduler.cancel(f"rem:{reminder.id}")
        await self._scheduler.cancel(f"rem:{reminder.id}:snooze")

    async def snooze(self, reminder: ReminderView, when: str = "") -> tuple[str, str | None]:
        """Перенести. Возвращает (текст-отклик, вопрос-переспрос | None)."""
        if when.strip():
            try:
                parsed = await self._parser.parse(when)
            except (WhenParseError, LLMError) as exc:
                log.warning("reminder_snooze_failed", when=when, error=str(exc))
                return (
                    f"не смог разобрать срок «{when}» — напоминание не перенесено",
                    None,
                )
            if parsed.kind == "unclear":
                return "", parsed.question
            due = parsed.due if parsed.due is not None else self._smart_moment()
        else:
            due = datetime.now(self._tz) + timedelta(minutes=self._snooze_minutes)
        if reminder.rrule:
            # серия продолжается по правилу; переносим только текущий раз
            await self._scheduler.schedule_once(
                REMINDER_TOPIC,
                {"reminder_id": reminder.id},
                at=due,
                job_id=f"rem:{reminder.id}:snooze",
            )
        else:
            await self._store.update(
                reminder.id, due=due.isoformat(), status="scheduled"
            )
            await self._scheduler.schedule_once(
                REMINDER_TOPIC,
                {"reminder_id": reminder.id},
                at=due,
                job_id=f"rem:{reminder.id}",
            )
        await self._audit.record(
            "reminder_snoozed", "reminders", f"{reminder.short_id} → {due.isoformat()}"
        )
        return f"⏰ Перенёс: «{reminder.text}» — напомню {self.format_due(due.isoformat())}.", None

    async def mark_done(self, reminder: ReminderView) -> str:
        """✅: закрыть напоминание; у повторяющихся серия живёт дальше;
        связанная открытая задача закрывается по-настоящему."""
        if reminder.task_id and self._tasks is not None:
            task = await self._tasks.get(reminder.task_id)
            if task is not None and task.status == "open":
                _, next_due = await self._tasks.complete(task)
                await self._audit.record("reminder_done", "reminders", reminder.short_id)
                # событие TaskChanged уже перенастроило авто-напоминание:
                # у повторяющейся задачи оно переехало на следующий срок —
                # тогда его не трогаем, иначе закрываем
                current = await self._store.get(reminder.id)
                if current is None or current.status != "scheduled":
                    await self._cancel_jobs(reminder)
                    await self._store.update(reminder.id, status="done")
                if next_due is not None:
                    return (
                        f"✅ Готово: «{reminder.text}». Задача повторяется — "
                        f"следующий срок {next_due}, напомню."
                    )
                return f"✅ Готово: «{reminder.text}». Задача закрыта."
        if reminder.rrule:
            await self._scheduler.cancel(f"rem:{reminder.id}:snooze")
            nxt = next_occurrence(
                reminder.rrule,
                datetime.fromisoformat(reminder.due),
                datetime.now(self._tz),
            )
            await self._audit.record("reminder_done", "reminders", reminder.short_id)
            if nxt is not None:
                await self._store.update(reminder.id, due=nxt.isoformat())
                return (
                    f"✅ Принято: «{reminder.text}». Повтор остаётся, следующее — "
                    f"{self.format_due(nxt.isoformat())}. Отключить совсем: ✖ или "
                    "«отмени напоминание»."
                )
        await self._cancel_jobs(reminder)
        await self._store.update(reminder.id, status="done")
        await self._audit.record("reminder_done", "reminders", reminder.short_id)
        return f"✅ Готово: «{reminder.text}»."

    async def cancel(self, reminder: ReminderView) -> str:
        await self._cancel_jobs(reminder)
        await self._store.update(reminder.id, status="cancelled")
        await self._audit.record("reminder_cancelled", "reminders", reminder.short_id)
        note = ""
        if reminder.task_id and self._tasks is not None:
            task = await self._tasks.get(reminder.task_id)
            if task is not None and task.status == "open":
                note = " Сама задача осталась открытой."
        series = " (вся серия)" if reminder.rrule else ""
        return f"✖ Отменил напоминание{series}: «{reminder.text}».{note}"

    async def resolve(self, ref: str) -> list[ReminderView]:
        return await self._store.resolve(self._owner, ref)

    async def list_active(self) -> list[ReminderView]:
        return await self._store.list_active(self._owner, self._list_limit)

    # ── срабатывание (событие JobFired от Scheduler) ─────────────────────────

    async def on_job_fired(self, event: JobFired) -> None:
        if event.topic != REMINDER_TOPIC:
            return
        reminder_id = event.payload.get("reminder_id", "")
        reminder = await self._store.get(reminder_id)
        if reminder is None or not reminder.active:
            log.info("reminder_fire_stale", reminder_id=reminder_id)
            return
        scheduled_iso = event.scheduled_for.isoformat()
        # связанное с задачей напоминание молчит, если задача уже не открыта
        if reminder.task_id and self._tasks is not None:
            task = await self._tasks.get(reminder.task_id)
            if task is None or task.status != "open":
                await self._store.try_log_delivery(
                    reminder.id, scheduled_iso, "skipped_task_closed"
                )
                await self._cancel_jobs(reminder)
                await self._store.update(reminder.id, status="done")
                log.info("reminder_skipped_task_closed", reminder_id=reminder.id)
                return
        if not await self._store.try_log_delivery(reminder.id, scheduled_iso, "delivered"):
            log.info("reminder_duplicate_suppressed", reminder_id=reminder.id)
            return
        await self._send(reminder, event.scheduled_for)
        await self._after_delivery(reminder, event.scheduled_for)

    async def _send(self, reminder: ReminderView, scheduled_for: datetime) -> None:
        lines = [f"🔔 Напоминание: {reminder.text}"]
        if reminder.task_id:
            lines.append(f"(по задаче #{reminder.task_id[:6]} — ✅ закроет её)")
        if reminder.rrule:
            lines.append(f"(повтор: {describe_rrule(reminder.rrule)})")
        late = datetime.now(self._tz) - scheduled_for.astimezone(self._tz)
        if late > LATE_NOTICE_AFTER:
            planned = self.format_due(scheduled_for.isoformat())
            lines.append(
                f"⚠️ Доставлено с опозданием (было назначено на {planned} — "
                "приложение в тот момент не работало)."
            )
        text = "\n".join(lines)
        actions = [
            MessageAction(id=f"rem:{ACTION_DONE}:{reminder.id}", label="✅ Сделал"),
            MessageAction(id=f"rem:{ACTION_SNOOZE}:{reminder.id}", label="⏰ Позже"),
            MessageAction(id=f"rem:{ACTION_CANCEL}:{reminder.id}", label="✖ Отменить"),
        ]
        for channel, user_id in self._targets:
            await self._deliver(
                OutgoingMessage(
                    user_id=user_id,
                    channel=channel,
                    text=text,
                    kind=OutgoingKind.NOTIFICATION,
                    actions=actions,
                )
            )
        await self._audit.record(
            "reminder_sent", "reminders", f"{reminder.short_id}: {reminder.text[:80]}"
        )

    async def _after_delivery(
        self, reminder: ReminderView, scheduled_for: datetime
    ) -> None:
        # следующий момент — строго после сработавшего (не только после «сейчас»:
        # при опоздавшей доставке серия всё равно должна продвинуться)
        after = max(datetime.now(self._tz), scheduled_for.astimezone(self._tz))
        if reminder.rrule:
            # серию ведёт Scheduler; обновляем срок для списка /reminders
            nxt = next_occurrence(
                reminder.rrule, datetime.fromisoformat(reminder.due), after
            )
            if nxt is not None:
                await self._store.update(reminder.id, due=nxt.isoformat())
            else:
                await self._store.update(reminder.id, status="fired")
            return
        if reminder.followup_minutes:
            # «если не сделал — напомню снова»: живёт до ✅/✖ (или закрытия задачи)
            nxt = datetime.now(self._tz) + timedelta(minutes=reminder.followup_minutes)
            await self._scheduler.schedule_once(
                REMINDER_TOPIC,
                {"reminder_id": reminder.id},
                at=nxt,
                job_id=f"rem:{reminder.id}",
            )
            await self._store.update(reminder.id, due=nxt.isoformat())
            return
        await self._store.update(reminder.id, status="fired")

    # ── кнопки (Router.handle_action, префикс «rem») ─────────────────────────

    async def handle_action(self, user_id: str, action_id: str) -> str:
        try:
            _, op, reminder_id = action_id.split(":", 2)
        except ValueError:
            return "Эта кнопка повреждена."
        reminder = await self._store.get(reminder_id)
        if reminder is None:
            return "Это напоминание уже не существует."
        await self._audit.record("reminder_action", "reminders", f"{op} {reminder.short_id}")
        if op == ACTION_DONE:
            if reminder.status == "done":
                return "Уже отмечено сделанным."
            if reminder.status == "cancelled":
                return "Это напоминание было отменено."
            return await self.mark_done(reminder)
        if op == ACTION_SNOOZE:
            if reminder.status == "cancelled":
                return "Это напоминание было отменено — создайте новое."
            reply, question = await self.snooze(reminder)
            return question or reply
        if op == ACTION_CANCEL:
            if reminder.status == "cancelled":
                return "Уже отменено."
            return await self.cancel(reminder)
        return "Неизвестное действие."

    # ── авто-напоминания задач (событие TaskChanged + стартовая синхронизация) ──

    async def on_task_changed(self, event: TaskChanged) -> None:
        await self._sync_task(
            event.task_id, event.title, event.status, event.due
        )

    async def _sync_task(
        self, task_id: str, title: str, status: str, due: str | None
    ) -> None:
        existing = await self._store.find_by_task(task_id)
        due_dt = datetime.fromisoformat(due) if due else None
        future = due_dt is not None and due_dt > datetime.now(self._tz)
        if status != "open" or not future:
            # задача закрыта/отменена, срок снят или уже в прошлом — молчим
            if existing is not None:
                await self._cancel_jobs(existing)
                await self._store.update(existing.id, status="cancelled")
                log.info("task_reminder_cancelled", task_id=task_id)
            return
        assert due is not None
        if existing is None:
            reminder = await self._store.add(
                self._owner, text=title, due=due, task_id=task_id
            )
            await self._schedule(reminder)
            log.info("task_reminder_created", task_id=task_id, due=due)
            return
        if existing.due != due or existing.text != title:
            updated = await self._store.update(existing.id, text=title, due=due)
            assert updated is not None
            await self._schedule(updated)
            log.info("task_reminder_rescheduled", task_id=task_id, due=due)

    async def sync_open_tasks(self) -> int:
        """Стартовая синхронизация: открытые задачи с будущим сроком получают
        авто-напоминания (в т.ч. созданные до Sprint 6). Возвращает число задач."""
        if self._tasks is None:
            return 0
        synced = 0
        for task in await self._tasks.search(status="open", limit=500):
            if task.due:
                await self._sync_task(task.id, task.title, task.status, task.due)
                synced += 1
        return synced

    # ── форматирование ───────────────────────────────────────────────────────

    def format_due(self, due_iso: str) -> str:
        due = datetime.fromisoformat(due_iso).astimezone(self._tz)
        now = datetime.now(self._tz)
        text = f"{WEEKDAY_SHORT[due.weekday()]} {due.day:02d}.{due.month:02d}"
        if due.year != now.year:
            text += f".{due.year}"
        return text + f" {due.hour:02d}:{due.minute:02d}"

    def format_reminder(self, reminder: ReminderView) -> str:
        parts = [self.format_due(reminder.due)]
        if reminder.rrule:
            parts.append(f"повтор: {describe_rrule(reminder.rrule)}")
        if reminder.followup_minutes:
            parts.append("повторяю, пока не сделано")
        if reminder.task_id:
            parts.append(f"по задаче #{reminder.task_id[:6]}")
        return f"#{reminder.short_id} {reminder.text} ({'; '.join(parts)})"

    async def overview_text(self) -> str:
        """Для /reminders: предстоящие напоминания, мимо LLM."""
        reminders = await self.list_active()
        if not reminders:
            return "Предстоящих напоминаний нет."
        total = await self._store.count_active(self._owner)
        lines = [f"Предстоящие напоминания ({total}):"]
        lines += [f"• {self.format_reminder(r)}" for r in reminders]
        if total > len(reminders):
            lines.append(f"… и ещё {total - len(reminders)}")
        return "\n".join(lines)

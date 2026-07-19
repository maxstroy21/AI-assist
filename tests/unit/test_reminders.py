"""Reminder Engine (Sprint 6): доставка, идемпотентность, snooze, follow-up,
авто-напоминания задач, кнопки."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sba.core.events import EventBus
from sba.core.types import OutgoingKind, OutgoingMessage
from sba.infra.audit import AuditLog
from sba.infra.db import Database
from sba.modules.reminders.service import ReminderService
from sba.modules.reminders.store import ReminderStore
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.scheduler.store import SchedulerStore
from sba.modules.tasks.dates import WhenParser
from sba.modules.tasks.events import TaskChanged
from sba.modules.tasks.service import TasksService
from sba.modules.tasks.store import TaskStore

TZ = ZoneInfo("Europe/Moscow")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


class Rig:
    """Собранный движок: шина, планировщик, задачи, доставленные уведомления."""

    def __init__(self, db: Database, followup_minutes: int = 60) -> None:
        self.bus = EventBus()
        self.deliveries: list[OutgoingMessage] = []
        self.scheduler = SchedulerService(SchedulerStore(db), self.bus, TZ)
        self.tasks = TasksService(
            TaskStore(db), WhenParser(None, TZ), TZ, bus=self.bus
        )

        async def deliver(out: OutgoingMessage) -> None:
            self.deliveries.append(out)

        self.store = ReminderStore(db)
        self.engine = ReminderService(
            self.store,
            self.scheduler,
            WhenParser(None, TZ),
            TZ,
            deliver=deliver,
            targets=[("cli", "local")],
            audit=AuditLog(db),
            tasks=self.tasks,
            followup_minutes=followup_minutes,
        )
        self.bus.subscribe(JobFired, self.engine.on_job_fired)
        self.bus.subscribe(TaskChanged, self.engine.on_task_changed)


@pytest.fixture
def rig(db: Database) -> Rig:
    return Rig(db)


def in_hours(hours: float) -> datetime:
    return datetime.now(UTC) + timedelta(hours=hours)


async def test_create_and_deliver_with_buttons(rig: Rig) -> None:
    outcome = await rig.engine.create("позвонить Ивану", when="через час")
    assert outcome.reminder is not None

    assert await rig.scheduler.tick(datetime.now(UTC)) == 0  # рано
    await rig.scheduler.tick(in_hours(2))
    assert len(rig.deliveries) == 1
    out = rig.deliveries[0]
    assert out.kind == OutgoingKind.NOTIFICATION
    assert "позвонить Ивану" in out.text
    assert [a.label for a in out.actions] == ["✅ Сделал", "⏰ Позже", "✖ Отменить"]

    await rig.scheduler.tick(in_hours(3))  # одноразовое: второй раз не приходит
    assert len(rig.deliveries) == 1
    updated = await rig.engine.resolve(outcome.reminder.id[:6])
    assert updated[0].status == "fired"


async def test_duplicate_fire_suppressed_by_log(rig: Rig) -> None:
    """Сбой между доставкой и продвижением джоба: повторное событие того же
    момента не шлёт дубль (reminder_log)."""
    outcome = await rig.engine.create("тест", when="через час")
    assert outcome.reminder is not None
    scheduled = datetime.fromisoformat(outcome.reminder.due)
    event = JobFired(
        job_id=f"rem:{outcome.reminder.id}",
        topic="reminder.fire",
        payload={"reminder_id": outcome.reminder.id},
        scheduled_for=scheduled,
    )
    await rig.engine.on_job_fired(event)
    await rig.engine.on_job_fired(event)  # «повтор после рестарта»
    assert len(rig.deliveries) == 1


async def test_smart_moment_without_when(rig: Rig) -> None:
    outcome = await rig.engine.create("проверить почту")
    assert outcome.reminder is not None
    due = datetime.fromisoformat(outcome.reminder.due)
    assert due > datetime.now(TZ)
    assert (due.hour, due.minute) == (9, 0)  # ближайшее утро в default_hour


async def test_unparsable_when_is_error_not_silent_guess(rig: Rig) -> None:
    outcome = await rig.engine.create("что-то", when="когда-нибудь потом")
    assert outcome.reminder is None
    assert outcome.error is not None and "не смог разобрать" in outcome.error


async def test_snooze_moves_delivery(rig: Rig) -> None:
    outcome = await rig.engine.create("тест", when="через час")
    assert outcome.reminder is not None
    reply, question = await rig.engine.snooze(outcome.reminder)  # без слов: +180 минут
    assert question is None and reply.startswith("⏰")

    await rig.scheduler.tick(in_hours(2))
    assert rig.deliveries == []  # старый срок уже не действует
    await rig.scheduler.tick(in_hours(4))
    assert len(rig.deliveries) == 1


async def test_cancel_stops_delivery(rig: Rig) -> None:
    outcome = await rig.engine.create("тест", when="через час")
    assert outcome.reminder is not None
    reply = await rig.engine.cancel(outcome.reminder)
    assert reply.startswith("✖")
    await rig.scheduler.tick(in_hours(2))
    assert rig.deliveries == []


async def test_followup_repeats_until_done(rig: Rig) -> None:
    """«Если не сделал — напомни снова»: повторы каждые followup_minutes (60),
    пока не нажато ✅."""
    outcome = await rig.engine.create("сдать отчёт", when="через час", repeat_until_done=True)
    assert outcome.reminder is not None
    rid = outcome.reminder.id

    await rig.scheduler.tick(in_hours(1.5))
    assert len(rig.deliveries) == 1
    await rig.scheduler.tick(in_hours(3))  # followup через 60 минут после доставки
    assert len(rig.deliveries) == 2

    ack = await rig.engine.handle_action("local", f"rem:done:{rid}")
    assert ack.startswith("✅")
    await rig.scheduler.tick(in_hours(10))
    assert len(rig.deliveries) == 2  # больше не приходит


async def test_recurring_reminder_continues(rig: Rig) -> None:
    outcome = await rig.engine.create("планёрка", when="каждый вторник в 10")
    assert outcome.reminder is not None
    assert outcome.reminder.rrule == "FREQ=WEEKLY;BYDAY=TU"
    first = datetime.fromisoformat(outcome.reminder.due)

    await rig.scheduler.tick(first + timedelta(minutes=1))
    assert len(rig.deliveries) == 1
    current = await rig.engine.resolve(outcome.reminder.id[:6])
    assert current[0].status == "scheduled"  # серия живёт
    assert datetime.fromisoformat(current[0].due) == first + timedelta(days=7)


async def test_task_with_due_gets_auto_reminder(rig: Rig) -> None:
    created = await rig.tasks.create("Позвонить Ивану", when="завтра в 9")
    assert created.task is not None
    reminder = await rig.engine.resolve("Позвонить Ивану")
    assert reminder and reminder[0].task_id == created.task.id
    assert reminder[0].due == created.task.due

    # закрытие задачи снимает напоминание
    await rig.tasks.complete(created.task)
    await rig.scheduler.tick(in_hours(48))
    assert rig.deliveries == []


async def test_task_due_change_reschedules_reminder(rig: Rig) -> None:
    created = await rig.tasks.create("Отчёт", when="завтра в 9")
    assert created.task is not None
    updated, _, _ = await rig.tasks.update(created.task, when="через 3 дня")
    reminder = await rig.engine.resolve("Отчёт")
    assert reminder[0].due == updated.due


async def test_fire_skips_when_task_closed_quietly(rig: Rig, db: Database) -> None:
    """Задача закрыта мимо событий (например, вручную в БД) — напоминание
    проверяет актуальность перед доставкой и молчит."""
    created = await rig.tasks.create("Отчёт", when="завтра в 9")
    assert created.task is not None
    reminder = (await rig.engine.resolve("Отчёт"))[0]
    await TaskStore(db).update("owner", created.task.id, status="done")

    await rig.scheduler.tick(in_hours(48))
    assert rig.deliveries == []
    refreshed = await rig.store.get(reminder.id)
    assert refreshed is not None and refreshed.status == "done"


async def test_done_button_closes_linked_task(rig: Rig) -> None:
    created = await rig.tasks.create("Отчёт", when="завтра в 9")
    assert created.task is not None
    await rig.scheduler.tick(in_hours(48))
    assert len(rig.deliveries) == 1
    reminder_id = rig.deliveries[0].actions[0].id.split(":", 2)[2]

    ack = await rig.engine.handle_action("local", f"rem:done:{reminder_id}")
    assert "закрыта" in ack.lower()
    task = await rig.tasks.get(created.task.id)
    assert task is not None and task.status == "done"


async def test_action_on_missing_reminder(rig: Rig) -> None:
    ack = await rig.engine.handle_action("local", "rem:done:deadbeef")
    assert "не существует" in ack


async def test_reminders_survive_restart(db: Database) -> None:
    rig1 = Rig(db)
    outcome = await rig1.engine.create("после рестарта", when="через час")
    assert outcome.reminder is not None

    rig2 = Rig(db)  # «новый процесс» над той же БД
    await rig2.scheduler.tick(in_hours(2))
    assert len(rig2.deliveries) == 1
    assert "после рестарта" in rig2.deliveries[0].text


async def test_sync_open_tasks_backfills_reminders(db: Database) -> None:
    """Задачи со сроком, созданные до Sprint 6 (без событий), получают
    напоминания при старте."""
    store = TaskStore(db)
    due = (datetime.now(TZ) + timedelta(days=1)).isoformat()
    await store.add("owner", "Старая задача", due=due)

    rig = Rig(db)
    assert await rig.engine.sync_open_tasks() == 1
    found = await rig.engine.resolve("Старая задача")
    assert found and found[0].due == due
    # повторный запуск не плодит дубликатов
    await rig.engine.sync_open_tasks()
    assert len(await rig.engine.resolve("Старая задача")) == 1


async def test_overview_text_lists_upcoming(rig: Rig) -> None:
    assert await rig.engine.overview_text() == "Предстоящих напоминаний нет."
    await rig.engine.create("первое", when="через час")
    text = await rig.engine.overview_text()
    assert text.startswith("Предстоящие напоминания") and "первое" in text

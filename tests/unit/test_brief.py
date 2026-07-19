"""Утренняя сводка (Sprint 6): содержимое, идемпотентность, misfire."""

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sba.core.events import EventBus
from sba.core.types import OutgoingMessage
from sba.infra.audit import AuditLog
from sba.infra.db import Database
from sba.modules.reminders.brief import BRIEF_JOB_ID, MorningBrief, parse_brief_time
from sba.modules.reminders.store import ReminderStore
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.scheduler.store import SchedulerStore
from sba.modules.tasks.dates import WhenParser
from sba.modules.tasks.service import TasksService
from sba.modules.tasks.store import TaskStore

TZ = ZoneInfo("Europe/Moscow")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


class Rig:
    def __init__(self, db: Database, grace_minutes: float = 240) -> None:
        self.bus = EventBus()
        self.deliveries: list[OutgoingMessage] = []
        self.scheduler = SchedulerService(
            SchedulerStore(db), self.bus, TZ, misfire_grace_minutes=grace_minutes
        )
        self.tasks = TasksService(TaskStore(db), WhenParser(None, TZ), TZ)
        self.reminder_store = ReminderStore(db)

        async def deliver(out: OutgoingMessage) -> None:
            self.deliveries.append(out)

        self.brief = MorningBrief(
            self.scheduler,
            self.reminder_store,
            TZ,
            deliver=deliver,
            targets=[("cli", "local")],
            audit=AuditLog(db),
            at=time(8, 30),
            tasks=self.tasks,
        )
        self.bus.subscribe(JobFired, self.brief.on_job_fired)


@pytest.fixture
def rig(db: Database) -> Rig:
    return Rig(db)


def test_parse_brief_time() -> None:
    assert parse_brief_time("08:30") == time(8, 30)
    assert parse_brief_time("23:05") == time(23, 5)


async def test_empty_day_text(rig: Rig) -> None:
    text = await rig.brief.build_text()
    assert text.startswith("🌅 Доброе утро!")
    assert "Свободный день" in text


async def test_brief_lists_today_overdue_and_reminders(rig: Rig, db: Database) -> None:
    now = datetime.now(TZ)
    store = TaskStore(db)
    await store.add(
        "owner", "Сегодняшняя задача",
        due=now.replace(hour=23, minute=50).isoformat(),
    )
    await store.add(
        "owner", "Просроченная", due=(now - timedelta(days=3)).isoformat()
    )
    await rig.reminder_store.add(
        "owner", "Напоминание дня", due=now.replace(hour=23, minute=55).isoformat()
    )
    # напоминание к задаче в сводке не дублируется (оно уже видно как задача)
    await rig.reminder_store.add(
        "owner", "Сегодняшняя задача",
        due=now.replace(hour=23, minute=50).isoformat(), task_id="t1",
    )

    text = await rig.brief.build_text()
    assert "Сегодняшняя задача" in text
    assert "23:50" in text
    assert "Просроченных задач: 1" in text
    assert "Напоминание дня" in text
    assert text.count("Сегодняшняя задача") == 1


async def test_schedule_keeps_missed_fire_for_grace_delivery(rig: Rig, db: Database) -> None:
    """Рестарт после пропущенного утра: schedule() не сдвигает созревший джоб,
    tick доставляет сводку в пределах grace."""
    await rig.brief.schedule()
    # смоделировать «вчера поставленный» джоб, созревший полчаса назад
    job = await rig.scheduler.get(BRIEF_JOB_ID)
    assert job is not None
    store = SchedulerStore(db)
    await store.upsert(
        BRIEF_JOB_ID, job.topic, job.payload,
        datetime.now(UTC) - timedelta(minutes=30),
        rrule=job.rrule, dtstart=job.dtstart, misfire=job.misfire,
    )
    await rig.brief.schedule()  # идемпотентно: не переписывает
    moved = await rig.scheduler.get(BRIEF_JOB_ID)
    assert moved is not None
    assert moved.next_fire_at < datetime.now(UTC)

    await rig.scheduler.tick(datetime.now(UTC))
    assert len(rig.deliveries) == 1
    assert rig.deliveries[0].text.startswith("🌅")


async def test_stale_brief_skipped_beyond_grace(db: Database) -> None:
    rig = Rig(db, grace_minutes=60)
    await rig.brief.schedule()
    job = await rig.scheduler.get(BRIEF_JOB_ID)
    assert job is not None
    await SchedulerStore(db).upsert(
        BRIEF_JOB_ID, job.topic, job.payload,
        datetime.now(UTC) - timedelta(hours=5),
        rrule=job.rrule, dtstart=job.dtstart, misfire=job.misfire,
    )
    await rig.scheduler.tick(datetime.now(UTC))
    assert rig.deliveries == []  # вечером утренняя сводка уже не нужна
    # джоб переведён на следующее утро
    advanced = await rig.scheduler.get(BRIEF_JOB_ID)
    assert advanced is not None and advanced.next_fire_at > datetime.now(UTC)


async def test_duplicate_brief_suppressed(rig: Rig) -> None:
    moment = datetime.now(UTC)
    event = JobFired(job_id=BRIEF_JOB_ID, topic="brief.morning", payload={},
                     scheduled_for=moment)
    await rig.brief.on_job_fired(event)
    await rig.brief.on_job_fired(event)
    assert len(rig.deliveries) == 1

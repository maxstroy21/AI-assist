"""Scheduler (Sprint 6): DB-джобы, misfire-политика, RRULE, рестарт."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sba.core.events import EventBus
from sba.infra.db import Database
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.scheduler.store import SchedulerStore

TZ = ZoneInfo("Europe/Moscow")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make(db: Database, bus: EventBus, grace_minutes: float = 240) -> SchedulerService:
    return SchedulerService(
        SchedulerStore(db), bus, TZ, misfire_grace_minutes=grace_minutes
    )


def collector(bus: EventBus) -> list[JobFired]:
    fired: list[JobFired] = []

    async def handler(event: JobFired) -> None:
        fired.append(event)

    bus.subscribe(JobFired, handler)
    return fired


async def test_one_shot_fires_once_and_disappears(db: Database) -> None:
    bus = EventBus()
    scheduler = make(db, bus)
    fired = collector(bus)
    now = datetime.now(UTC)
    await scheduler.schedule_once(
        "reminder.fire", {"reminder_id": "r1"}, now - timedelta(minutes=1)
    )

    assert await scheduler.tick(now) == 1
    assert fired[0].topic == "reminder.fire"
    assert fired[0].payload == {"reminder_id": "r1"}
    assert await scheduler.tick(now) == 0  # одноразовый джоб исчез


async def test_future_job_does_not_fire(db: Database) -> None:
    bus = EventBus()
    scheduler = make(db, bus)
    fired = collector(bus)
    now = datetime.now(UTC)
    await scheduler.schedule_once("t", {}, now + timedelta(hours=1))
    assert await scheduler.tick(now) == 0
    assert fired == []


async def test_rrule_job_advances_to_next_occurrence(db: Database) -> None:
    bus = EventBus()
    scheduler = make(db, bus)
    fired = collector(bus)
    now = datetime.now(TZ)
    dtstart = now.replace(hour=9, minute=0, second=0, microsecond=0)
    job_id = await scheduler.schedule_rrule("t", {}, "FREQ=DAILY", dtstart)
    assert job_id is not None
    job = await scheduler.get(job_id)
    assert job is not None

    first = job.next_fire_at
    assert await scheduler.tick(first + timedelta(minutes=1)) == 1
    advanced = await scheduler.get(job_id)
    assert advanced is not None
    assert advanced.next_fire_at == first + timedelta(days=1)
    assert len(fired) == 1


async def test_misfire_skip_drops_stale_fire(db: Database) -> None:
    """skip-джоб (сводка), пропущенный дольше grace, не сваливается вечером."""
    bus = EventBus()
    scheduler = make(db, bus, grace_minutes=60)
    fired = collector(bus)
    now = datetime.now(UTC)
    await scheduler.schedule_once("brief", {}, now - timedelta(hours=5), misfire="skip")
    assert await scheduler.tick(now) == 0
    assert fired == []
    # в пределах grace — доставляется
    await scheduler.schedule_once("brief", {}, now - timedelta(minutes=30), misfire="skip")
    assert await scheduler.tick(now) == 1


async def test_misfire_deliver_fires_late(db: Database) -> None:
    """Напоминание доставляется даже сильно позже срока (лучше поздно)."""
    bus = EventBus()
    scheduler = make(db, bus, grace_minutes=60)
    fired = collector(bus)
    now = datetime.now(UTC)
    await scheduler.schedule_once("reminder.fire", {}, now - timedelta(days=2))
    assert await scheduler.tick(now) == 1
    assert len(fired) == 1


async def test_handler_reschedule_survives_advance(db: Database) -> None:
    """Обработчик перепланировал джоб тем же id (follow-up) — продвижение
    одноразового джоба не должно снести новую версию."""
    bus = EventBus()
    scheduler = make(db, bus)
    now = datetime.now(UTC)
    future = now + timedelta(hours=2)

    async def reschedule(event: JobFired) -> None:
        await scheduler.schedule_once(event.topic, event.payload, future, job_id=event.job_id)

    bus.subscribe(JobFired, reschedule)
    await scheduler.schedule_once("t", {"x": "1"}, now - timedelta(minutes=1), job_id="j1")
    await scheduler.tick(now)
    job = await scheduler.get("j1")
    assert job is not None and job.next_fire_at == future


async def test_jobs_survive_restart(db: Database) -> None:
    """«Рестарт»: новый экземпляр сервиса над той же БД видит джоб."""
    bus1 = EventBus()
    await make(db, bus1).schedule_once("t", {"a": "b"}, datetime.now(UTC) - timedelta(minutes=1))

    bus2 = EventBus()
    scheduler2 = make(db, bus2)
    fired = collector(bus2)
    assert await scheduler2.tick(datetime.now(UTC)) == 1
    assert fired[0].payload == {"a": "b"}


async def test_cancel_removes_job(db: Database) -> None:
    bus = EventBus()
    scheduler = make(db, bus)
    fired = collector(bus)
    now = datetime.now(UTC)
    job_id = await scheduler.schedule_once("t", {}, now - timedelta(minutes=1))
    await scheduler.cancel(job_id)
    assert await scheduler.tick(now) == 0
    assert fired == []

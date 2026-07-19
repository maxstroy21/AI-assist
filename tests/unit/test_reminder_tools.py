"""Инструменты напоминаний: линия недоверия к модели (Sprint 6)."""

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from sba.core.events import EventBus
from sba.core.types import OutgoingMessage
from sba.infra.audit import AuditLog
from sba.infra.db import Database
from sba.modules.reminders.service import ReminderService
from sba.modules.reminders.store import ReminderStore
from sba.modules.reminders.tools import build_reminder_tools
from sba.modules.scheduler.service import SchedulerService
from sba.modules.scheduler.store import SchedulerStore
from sba.modules.tasks.dates import WhenParser

TZ = ZoneInfo("Europe/Moscow")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


@pytest.fixture
def service(db: Database) -> ReminderService:
    async def deliver(out: OutgoingMessage) -> None:  # pragma: no cover
        pass

    return ReminderService(
        ReminderStore(db),
        SchedulerService(SchedulerStore(db), EventBus(), TZ),
        WhenParser(None, TZ),
        TZ,
        deliver=deliver,
        targets=[],
        audit=AuditLog(db),
    )


async def call(service: ReminderService, name: str, **kwargs: object) -> str:
    spec = next(s for s in build_reminder_tools(service) if s.name == name)
    args: BaseModel = spec.args_schema(**kwargs)
    return await spec.handler(args)


async def test_create_list_snooze_cancel_flow(service: ReminderService) -> None:
    created = await call(
        service, "create_reminder", text="позвонить Ивану", when="завтра в 15:00"
    )
    assert "Создал напоминание" in created and "15:00" in created

    listed = await call(service, "list_reminders")
    assert "позвонить Ивану" in listed and "id" in listed

    snoozed = await call(service, "snooze_reminder", reminder="позвонить ивану",
                         when="послезавтра в 10")
    assert snoozed.startswith("⏰") and "10:00" in snoozed

    cancelled = await call(service, "cancel_reminder", reminder="позвонить ивану")
    assert cancelled.startswith("✖")
    assert "нет" in (await call(service, "list_reminders")).lower()


async def test_create_without_when_notes_smart_moment(service: ReminderService) -> None:
    text = await call(service, "create_reminder", text="проверить почту")
    assert "ближайшее утро" in text


async def test_empty_list_tells_model_not_to_invent(service: ReminderService) -> None:
    listed = await call(service, "list_reminders")
    assert "не выдумывай" in listed


async def test_missing_reference_points_to_list(service: ReminderService) -> None:
    reply = await call(service, "cancel_reminder", reminder="нет такого")
    assert "не найдено" in reply and "list_reminders" in reply


async def test_ambiguous_reference_asks_for_id(service: ReminderService) -> None:
    await call(service, "create_reminder", text="оплатить хостинг", when="завтра в 10")
    await call(service, "create_reminder", text="оплатить интернет", when="завтра в 11")
    reply = await call(service, "snooze_reminder", reminder="оплатить")
    assert "уточни по id" in reply

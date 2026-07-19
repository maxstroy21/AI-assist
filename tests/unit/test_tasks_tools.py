"""Сервис и инструменты задач: сценарии владельца (Sprint 5)."""

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel

from sba.core import execctx
from sba.core.tools.spec import ToolSpec
from sba.infra.db import Database
from sba.llm.providers.fake import FakeLLM
from sba.modules.tasks.dates import WhenParser
from sba.modules.tasks.service import TasksService
from sba.modules.tasks.store import TaskStore
from sba.modules.tasks.tools import build_task_tools

TZ = ZoneInfo("Europe/Moscow")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_service(db: Database, llm: FakeLLM | None = None) -> TasksService:
    return TasksService(TaskStore(db), WhenParser(llm, TZ), TZ)


def tool(service: TasksService, name: str) -> ToolSpec:
    return next(s for s in build_task_tools(service) if s.name == name)


async def call(service: TasksService, name: str, **kwargs: object) -> str:
    spec = tool(service, name)
    args: BaseModel = spec.args_schema(**kwargs)
    return await spec.handler(args)


async def test_create_search_complete_flow(db: Database) -> None:
    service = make_service(db)
    created = await call(
        service, "create_task",
        title="Позвонить Ивану", when="завтра в 15:00", project="Экспедиция",
    )
    assert "Создал задачу" in created and "Позвонить Ивану" in created
    assert "срок:" in created and "15:00" in created

    listed = await call(service, "search_tasks", project="экспедиция")
    assert "Позвонить Ивану" in listed

    done = await call(service, "complete_task", task="позвонить ивану")
    assert done.startswith("✅")
    empty = await call(service, "search_tasks")
    assert empty.startswith("Задач")  # открытых больше нет


async def test_unclear_when_asks_instead_of_creating(db: Database) -> None:
    llm = FakeLLM(
        ['{"kind": "unclear", "due": null, "confidence": 0.9,'
         ' "question": "В этот понедельник или в следующий?"}']
    )
    service = make_service(db, llm)
    reply = await call(service, "create_task", title="Сдать отчёт", when="как договаривались")
    assert "НЕ создана" in reply
    assert "В этот понедельник или в следующий?" in reply
    assert await service.count_open() == 0  # DoD: переспрос вместо угадывания


async def test_parse_failure_creates_task_without_due(db: Database) -> None:
    service = make_service(db, llm=None)  # роль extraction не настроена
    reply = await call(service, "create_task", title="Разобрать гараж", when="когда-нибудь потом")
    assert "Создал задачу" in reply
    assert "⚠️" in reply and "без срока" in reply
    assert await service.count_open() == 1  # CRUD живёт и без LLM


async def test_recurring_complete_rolls_forward(db: Database) -> None:
    service = make_service(db)
    await call(service, "create_task", title="Полить цветы", when="каждый понедельник в 9")
    reply = await call(service, "complete_task", task="полить цветы")
    assert "повторяется" in reply and "следующий срок" in reply
    # задача осталась открытой, срок сдвинулся на будущий понедельник
    [task] = await service.search()
    assert task.status == "open"
    assert task.due is not None
    assert datetime.fromisoformat(task.due) > datetime.now(TZ)


async def test_complete_ambiguous_lists_candidates(db: Database) -> None:
    service = make_service(db)
    await call(service, "create_task", title="Оплатить хостинг")
    await call(service, "create_task", title="Оплатить интернет")
    reply = await call(service, "complete_task", task="оплатить")
    assert "уточни" in reply.lower()
    assert "хостинг" in reply and "интернет" in reply
    assert await service.count_open() == 2  # ничего не закрыли молча


async def test_complete_unknown_is_honest(db: Database) -> None:
    service = make_service(db)
    reply = await call(service, "complete_task", task="несуществующая")
    assert "не найдена" in reply


async def test_update_moves_due_and_cancels(db: Database) -> None:
    service = make_service(db)
    await call(service, "create_task", title="Сдать отчёт", when="завтра")
    reply = await call(service, "update_task", task="сдать отчёт", when="через 3 дня")
    assert "Обновил задачу" in reply
    [task] = await service.search()
    expected = (datetime.now(TZ) + timedelta(days=3)).date()
    assert task.due is not None
    assert datetime.fromisoformat(task.due).date() == expected

    reply = await call(service, "update_task", task="сдать отчёт", status="cancelled")
    assert "отменена" in reply
    assert await service.count_open() == 0


async def test_update_removes_due(db: Database) -> None:
    service = make_service(db)
    await call(service, "create_task", title="Прочитать книгу", when="завтра")
    await call(service, "update_task", task="прочитать книгу", when="без срока")
    [task] = await service.search()
    assert task.due is None and task.rrule is None


async def test_source_message_id_from_execctx(db: Database) -> None:
    service = make_service(db)
    token = execctx.current_message_id.set("msg-42")
    try:
        await call(service, "create_task", title="Из диалога")
    finally:
        execctx.current_message_id.reset(token)
    [task] = await service.search()
    assert task.source_message_id == "msg-42"  # DoD: связь задача ↔ сообщение


async def test_overview_text_for_service_command(db: Database) -> None:
    service = make_service(db)
    assert await service.overview_text() == "Открытых задач нет."
    await call(service, "create_task", title="Купить палатку", when="25.12")
    text = await service.overview_text()
    assert text.startswith("Открытые задачи (1):")
    assert "Купить палатку" in text and "#" in text


async def test_overdue_is_marked(db: Database) -> None:
    service = make_service(db)
    store = TaskStore(db)
    past = (datetime.now(TZ) - timedelta(days=2)).isoformat()
    await store.add("owner", "Просроченная", due=past)
    text = await service.overview_text()
    assert "просрочена" in text

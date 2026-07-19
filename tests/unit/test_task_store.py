"""Хранилище задач: CRUD, FTS-поиск, разрешение ссылок (Sprint 5)."""

from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.modules.tasks.store import TaskStore


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


@pytest.fixture
def store(db: Database) -> TaskStore:
    return TaskStore(db)


async def test_add_and_search_russian_morphology(store: TaskStore) -> None:
    await store.add("owner", "Позвонить Ивану насчёт сметы", project="Экспедиция")
    found = await store.search("owner", query="звонок Ивану")
    assert len(found) == 1
    assert found[0].title == "Позвонить Ивану насчёт сметы"


async def test_search_filters_by_project_and_status(store: TaskStore) -> None:
    a = await store.add("owner", "Смета", project="Экспедиция-2026")
    await store.add("owner", "Отчёт", project="Работа")
    by_project = await store.search("owner", project="экспедиция")
    assert [t.id for t in by_project] == [a.id]

    await store.update("owner", a.id, status="done")
    assert await store.search("owner", project="экспедиция") == []
    done = await store.search("owner", project="экспедиция", status="done")
    assert [t.id for t in done] == [a.id]


async def test_list_orders_by_due_then_no_due(store: TaskStore) -> None:
    no_due = await store.add("owner", "Без срока")
    late = await store.add("owner", "Позже", due="2026-07-25T09:00:00+03:00")
    soon = await store.add("owner", "Скоро", due="2026-07-20T09:00:00+03:00")
    tasks = await store.search("owner")
    assert [t.id for t in tasks] == [soon.id, late.id, no_due.id]


async def test_update_reindexes_fts(store: TaskStore) -> None:
    task = await store.add("owner", "Старое название")
    await store.update("owner", task.id, title="Купить палатку")
    assert await store.search("owner", query="старое название") == []
    found = await store.search("owner", query="палатка")
    assert [t.id for t in found] == [task.id]


async def test_update_due_and_clear_due(store: TaskStore) -> None:
    task = await store.add("owner", "Задача", due="2026-07-20T09:00:00+03:00", rrule=None)
    updated = await store.update(
        "owner", task.id, due="2026-07-21T10:00:00+03:00", rrule="FREQ=DAILY"
    )
    assert updated is not None
    assert updated.due == "2026-07-21T10:00:00+03:00"
    assert updated.rrule == "FREQ=DAILY"
    cleared = await store.update("owner", task.id, clear_due=True)
    assert cleared is not None
    assert cleared.due is None and cleared.rrule is None


async def test_resolve_by_short_id_and_title(store: TaskStore) -> None:
    task = await store.add("owner", "Оплатить хостинг")
    await store.add("owner", "Другая задача")
    by_id = await store.resolve("owner", f"#{task.id[:6]}")
    assert [t.id for t in by_id] == [task.id]
    by_title = await store.resolve("owner", "хостинг")
    assert [t.id for t in by_title] == [task.id]


async def test_resolve_falls_back_to_substring(store: TaskStore) -> None:
    task = await store.add("owner", "Сдать ГТО")  # короткие слова мимо FTS-префиксов
    found = await store.resolve("owner", "гто")
    assert [t.id for t in found] == [task.id]


async def test_source_message_link_is_stored(store: TaskStore) -> None:
    task = await store.add("owner", "Из сообщения", source_message_id="msg-123")
    loaded = await store.get("owner", task.id)
    assert loaded is not None
    assert loaded.source_message_id == "msg-123"


async def test_other_user_tasks_invisible(store: TaskStore) -> None:
    await store.add("someone", "Чужая задача")
    assert await store.search("owner") == []
    assert await store.resolve("owner", "чужая") == []

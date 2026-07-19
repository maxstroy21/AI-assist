from pathlib import Path

import pytest
from pydantic import BaseModel

from sba.infra.db import Database
from sba.modules.memory.episodes import EpisodeStore
from sba.modules.memory.service import MemoryService
from sba.modules.memory.store import MemoryStore
from sba.modules.memory.tools import (
    ForgetArgs,
    RecallArgs,
    RememberArgs,
    build_memory_tools,
)


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


@pytest.fixture
def service(db: Database) -> MemoryService:
    return MemoryService(MemoryStore(db), episodes=EpisodeStore(db))


def get_handler(service: MemoryService, name: str):
    specs = {s.name: s for s in build_memory_tools(service)}
    return specs[name].handler


async def test_remember_recall_forget_cycle(service: MemoryService) -> None:
    remember = get_handler(service, "remember_fact")
    recall = get_handler(service, "recall_memory")
    forget = get_handler(service, "forget_memory")

    result = await remember(
        RememberArgs(type="person", subject="Иван Петров", content="подрядчик по смете")
    )
    assert "Запомнил" in result

    found = await recall(RecallArgs(query="кто такой Иван?"))
    assert "Иван Петров" in found and "подрядчик" in found

    forgotten = await forget(ForgetArgs(query="Иван Петров"))
    assert "Забыто" in forgotten

    after = await recall(RecallArgs(query="кто такой Иван?"))
    assert "ничего не найдено" in after


async def test_preferences_text_for_prompt(service: MemoryService) -> None:
    assert await service.preferences_text() is None
    await service.remember("preference", "стиль ответов", "отвечать кратко")
    text = await service.preferences_text()
    assert text is not None
    assert "кратко" in text


async def test_relevant_facts_exclude_preferences(service: MemoryService) -> None:
    await service.remember("preference", "стиль ответов", "кратко")
    await service.remember("project", "Экспедиция", "старт в июле")
    facts = await service.relevant_facts_text("что по проекту Экспедиция?")
    assert facts is not None
    assert "Экспедиция" in facts
    assert "кратко" not in facts  # предпочтения идут в system, не в факты


async def test_forget_nothing_found_message(service: MemoryService) -> None:
    forget = get_handler(service, "forget_memory")
    result = await forget(ForgetArgs(query="несуществующая тема"))
    assert "нечего забывать" in result


# ── Sprint 7: агрегированный recall и ревизия ────────────────────────────────


async def test_recall_aggregates_facts_history_and_episodes(
    db: Database, service: MemoryService
) -> None:
    await service.remember("decision", "выбор подрядчика", "работаем с ООО Ромашка")
    await service.remember("decision", "выбор подрядчика", "передумали: ИП Иванов")
    await EpisodeStore(db).add(
        "owner", "conv1", "Обсуждали выбор подрядчика для экспедиции.",
        ["подрядчик"], None, "2026-07-10T10:00:00+00:00", "2026-07-10T11:00:00+00:00",
    )
    recall = get_handler(service, "recall_memory")
    text = await recall(RecallArgs(query="что ты знаешь о выборе подрядчика?"))
    assert "ИП Иванов" in text                    # актуальное решение
    assert "ранее" in text and "Ромашка" in text  # история вытеснения видна
    assert "Из прошлых разговоров" in text and "2026-07-10" in text


async def test_recall_project_filter(service: MemoryService) -> None:
    store = service._store  # юнит-тест внутренней связки
    await store.add("owner", "fact", "снаряжение", "нужна палатка", project="Экспедиция")
    await store.add("owner", "fact", "снаряжение", "рюкзак порвался")
    recall = get_handler(service, "recall_memory")
    text = await recall(RecallArgs(query="снаряжение", project="Экспедиция"))
    assert "палатка" in text and "рюкзак" not in text


async def test_review_text_marks_auto_facts(db: Database, service: MemoryService) -> None:
    await service.remember("fact", "ручное", "записано явно")
    store = MemoryStore(db)
    await store.add("owner", "fact", "авто", "извлечено из разговора", source="auto:conv1")
    text = await service.review_text()
    assert "активных фактов — 2" in text
    assert "🤖" in text and "✍️" in text
    assert "забудь" in text


async def test_review_text_empty_memory(service: MemoryService) -> None:
    text = await service.review_text()
    assert "новых фактов не появилось" in text


async def test_remember_args_validation() -> None:
    with pytest.raises(ValueError):
        RememberArgs(type="unknown", subject="x", content="y")  # type: ignore[arg-type]


def test_tools_registered_with_expected_risks(tmp_path: Path) -> None:
    class Dummy(BaseModel):
        pass

    # build_memory_tools не требует живой БД для декларации
    service = MemoryService.__new__(MemoryService)  # type: ignore[call-arg]
    specs = {s.name: s.risk for s in build_memory_tools(service)}
    assert specs == {
        "remember_fact": "write",
        "recall_memory": "read",
        "forget_memory": "write",
    }

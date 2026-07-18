from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.modules.memory.store import MemoryStore, build_fts_query


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


@pytest.fixture
def store(db: Database) -> MemoryStore:
    return MemoryStore(db)


def test_fts_query_builds_prefixes() -> None:
    query = build_fts_query("что ты знаешь про Ивана Петрова?")
    assert query is not None
    assert '"иван"*' in query
    assert '"петро"*' in query
    assert "знаешь" not in query  # стоп-слово


def test_fts_query_none_for_stopwords_only() -> None:
    assert build_fts_query("что ты помнишь?") is None


async def test_add_and_search_with_russian_morphology(store: MemoryStore) -> None:
    await store.add("owner", "person", "Иван Петров", "подрядчик по смете экспедиции")
    found = await store.search("owner", "что известно про Ивана?")
    assert len(found) == 1
    assert found[0].subject == "Иван Петров"


async def test_search_by_content_words(store: MemoryStore) -> None:
    await store.add("owner", "decision", "выбор подрядчика", "решили работать с ООО Ромашка")
    found = await store.search("owner", "какое решение по подрядчику приняли?")
    assert found and "Ромашка" in found[0].content


async def test_memory_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "persist.db"
    db1 = await Database.open(path)
    await MemoryStore(db1).add("owner", "fact", "день рождения мамы", "3 марта")
    await db1.close()

    db2 = await Database.open(path)
    found = await MemoryStore(db2).search("owner", "когда день рождения мамы?")
    await db2.close()
    assert found and "3 марта" in found[0].content


async def test_retract_hides_fact(store: MemoryStore) -> None:
    await store.add("owner", "fact", "старый пароль от роутера", "qwerty123")
    forgotten = await store.retract("owner", "пароль роутера")
    assert len(forgotten) == 1
    assert await store.search("owner", "пароль роутера") == []


async def test_retract_without_specific_query_refused(store: MemoryStore) -> None:
    await store.add("owner", "fact", "тема", "содержимое")
    assert await store.retract("owner", "всё это") == []


async def test_preference_superseded_by_same_subject(store: MemoryStore) -> None:
    await store.add("owner", "preference", "стиль ответов", "отвечать подробно")
    await store.add("owner", "preference", "стиль ответов", "отвечать кратко")
    preferences = await store.preferences("owner")
    assert len(preferences) == 1
    assert preferences[0].content == "отвечать кратко"


async def test_preferences_of_different_subjects_coexist(store: MemoryStore) -> None:
    await store.add("owner", "preference", "стиль ответов", "кратко")
    await store.add("owner", "preference", "язык", "русский")
    assert len(await store.preferences("owner")) == 2


async def test_empty_query_returns_recent(store: MemoryStore) -> None:
    await store.add("owner", "fact", "первое", "раз")
    await store.add("owner", "fact", "второе", "два")
    found = await store.search("owner", "что ты помнишь?")
    assert len(found) == 2

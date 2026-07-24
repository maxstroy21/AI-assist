import asyncio
from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.infra.text import build_fts_query
from sba.modules.memory.store import MemoryStore


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


class _SlowEmbedder:
    """Эмулирует холодную bge-m3 на CPU: эмбеддинг медленнее потолка recall."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0.2)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _NoopVectors:
    async def ensure_collection(self, *a, **k) -> None: ...
    async def upsert(self, *a, **k) -> None: ...
    async def delete(self, *a, **k) -> None: ...

    async def search(self, *a, **k):  # до сюда не доходим — эмбеддинг таймаутит
        raise AssertionError("векторный поиск не должен вызываться при таймауте")


async def test_slow_semantic_recall_falls_back_to_lexical(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """recall не виснет на холодной bge-m3: семантическая половина под таймаутом,
    при превышении отдаёт быстрые лексические результаты (живая проверка: recall
    «думал» 60 с на загрузке эмбеддинг-модели)."""
    monkeypatch.setattr("sba.modules.memory.store.VECTOR_SEARCH_TIMEOUT", 0.05)
    store = MemoryStore(db, vectors=_NoopVectors(), embedder=_SlowEmbedder())
    await store.add("owner", "project", "статус ППЭЭ", "проект ППЭЭ в активной фазе")
    found = await store.search("owner", "какой статус у ППЭЭ")
    assert found and "ППЭЭ" in found[0].subject  # лексический результат получен


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


# ── Sprint 7: решения, проекты, история ──────────────────────────────────────


async def test_decision_superseded_by_same_subject(store: MemoryStore) -> None:
    await store.add("owner", "decision", "выбор подрядчика", "работаем с ООО Ромашка")
    await store.add("owner", "decision", "выбор подрядчика", "передумали: ИП Иванов")
    active = await store.search("owner", "решение по подрядчику")
    assert len(active) == 1
    assert "Иванов" in active[0].content


async def test_history_returns_superseded_chain(store: MemoryStore) -> None:
    first = await store.add("owner", "decision", "маршрут", "идём через перевал")
    second = await store.add("owner", "decision", "маршрут", "решили в обход")
    third = await store.add("owner", "decision", "маршрут", "вернулись к перевалу")
    chain = await store.history(third.id)
    assert [f.id for f in chain] == [second.id, first.id]


async def test_plain_facts_same_subject_accumulate(store: MemoryStore) -> None:
    await store.add("owner", "person", "Иван Петров", "подрядчик по смете")
    await store.add("owner", "person", "Иван Петров", "живёт в Твери")
    found = await store.search("owner", "Иван Петров", k=10)
    assert len(found) == 2  # факты о человеке копятся, не вытесняются


async def test_search_filters_by_project(store: MemoryStore) -> None:
    await store.add(
        "owner", "fact", "снаряжение", "нужна новая палатка", project="Экспедиция"
    )
    await store.add("owner", "fact", "снаряжение", "старый рюкзак порвался")
    all_facts = await store.search("owner", "снаряжение", k=10)
    assert len(all_facts) == 2
    scoped = await store.search("owner", "снаряжение", k=10, project="экспедиция")
    assert len(scoped) == 1
    assert scoped[0].project == "Экспедиция"


async def test_recent_lists_fresh_facts_with_source(store: MemoryStore) -> None:
    await store.add("owner", "fact", "тема", "содержимое", source="auto:abc12345")
    recent = await store.recent("owner", days=7)
    assert len(recent) == 1
    assert recent[0].source == "auto:abc12345"

"""Эпизодическая память (Sprint 7): хранение и поиск итогов разговоров."""

from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.modules.memory.episodes import EpisodeStore


@pytest.fixture
async def store(tmp_path: Path) -> EpisodeStore:
    db = await Database.open(tmp_path / "test.db")
    yield EpisodeStore(db)
    await db.close()


async def test_add_and_search(store: EpisodeStore) -> None:
    episode = await store.add(
        "owner", "conv1", "Обсуждали смету экспедиции и подрядчика.",
        ["смета", "экспедиция"], "Экспедиция-2026",
        "2026-07-10T10:00:00+00:00", "2026-07-10T11:00:00+00:00",
    )
    assert episode is not None
    found = await store.search("owner", "о чём говорили про смету?")
    assert len(found) == 1
    assert found[0].project == "Экспедиция-2026"


async def test_one_episode_per_conversation(store: EpisodeStore) -> None:
    args = ("owner", "conv1", "Итог разговора про рыбалку.", ["рыбалка"], None,
            "2026-07-10T10:00:00+00:00", "2026-07-10T11:00:00+00:00")
    assert await store.add(*args) is not None
    assert await store.add(*args) is None  # повторная консолидация не дублирует
    assert await store.count("owner") == 1


async def test_search_by_topic_words(store: EpisodeStore) -> None:
    await store.add(
        "owner", "conv1", "Короткий итог.", ["датчики давления"], None,
        "2026-07-10T10:00:00+00:00", "2026-07-10T11:00:00+00:00",
    )
    found = await store.search("owner", "что известно про датчики?")
    assert len(found) == 1


async def test_stopword_query_returns_nothing(store: EpisodeStore) -> None:
    await store.add(
        "owner", "conv1", "Итог.", [], None,
        "2026-07-10T10:00:00+00:00", "2026-07-10T11:00:00+00:00",
    )
    assert await store.search("owner", "что ты?") == []

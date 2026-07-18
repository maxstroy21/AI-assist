"""Векторный recall памяти: гибрид с FTS, деградация без эмбеддера."""

from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.infra.vectors import VectorStore
from sba.llm.gateway import LLMError
from sba.modules.memory.store import MemoryStore


class SynonymEmbedder:
    """«пёс» и «собака» — одна ось; иначе — ортогональные векторы."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0 if ("пёс" in t.lower() or "собак" in t.lower()) else 0.0, 0.1]
            for t in texts
        ]


class BrokenEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError("модель недоступна")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


async def test_semantic_recall_beyond_fts(db: Database) -> None:
    store = MemoryStore(db, vectors=VectorStore.in_memory(), embedder=SynonymEmbedder())
    await store.add("owner", "fact", "Пёс Рекс", "охраняет дачу по выходным")
    found = await store.search("owner", "что известно о собаке?")
    assert any(f.subject == "Пёс Рекс" for f in found)  # FTS это слово не нашла бы


async def test_retracted_fact_gone_from_vector_recall(db: Database) -> None:
    store = MemoryStore(db, vectors=VectorStore.in_memory(), embedder=SynonymEmbedder())
    await store.add("owner", "fact", "Пёс Рекс", "охраняет дачу")
    await store.retract("owner", "пёс Рекс")
    assert await store.search("owner", "что известно о собаке?") == []


async def test_memory_works_without_embedder(db: Database) -> None:
    """Ollama выключена → «запомни/вспомни» продолжают работать на FTS."""
    store = MemoryStore(db, vectors=VectorStore.in_memory(), embedder=BrokenEmbedder())
    fact = await store.add("owner", "person", "Иван Петров", "подрядчик по смете")
    assert fact.id
    found = await store.search("owner", "кто такой Иван?")
    assert found and found[0].subject == "Иван Петров"


async def test_superseded_preference_leaves_vector_index(db: Database) -> None:
    store = MemoryStore(db, vectors=VectorStore.in_memory(), embedder=SynonymEmbedder())
    await store.add("owner", "preference", "стиль ответов", "отвечай про собак подробно")
    await store.add("owner", "preference", "стиль ответов", "отвечай кратко")
    found = await store.search("owner", "про собаку")
    # вытеснённое предпочтение не должно всплывать даже семантикой
    assert all("подробно" not in f.content for f in found)

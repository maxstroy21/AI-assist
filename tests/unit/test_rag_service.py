"""RAG Service: гибридный поиск (FTS + вектор, RRF), замена и удаление документов."""

from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.infra.vectors import VectorStore
from sba.llm.gateway import LLMError
from sba.modules.indexer.store import CatalogStore
from sba.modules.rag.interface import Chunk
from sba.modules.rag.service import RAGService
from sba.modules.rag.store import ChunkStore
from sba.modules.rag.tools import build_rag_tools


class FakeEmbedder:
    """Детерминированные векторы: ось на ключевое слово + константная ось
    (чтобы вектор никогда не был нулевым)."""

    def __init__(self, axes: list[str]) -> None:
        self.axes = axes
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [
            [1.0 if axis in text.lower() else 0.0 for axis in self.axes] + [0.1]
            for text in texts
        ]


class BrokenEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise LLMError("модель недоступна")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


async def register_file(db: Database, path: str) -> str:
    record = await CatalogStore(db).upsert_pending(path, "hash", 1.0, 10)
    return record.id


def make_service(db: Database, embedder: object) -> RAGService:
    return RAGService(ChunkStore(db), VectorStore.in_memory(), embedder, embed_batch=2)  # type: ignore[arg-type]


class SynonymEmbedder:
    """«пёс» и «собака» — одна семантическая ось (лексически не совпадают)."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0 if ("пёс" in t.lower() or "собак" in t.lower()) else 0.0, 0.1]
            for t in texts
        ]


async def test_hybrid_search_combines_semantic_and_lexical(db: Database) -> None:
    service = make_service(db, SynonymEmbedder())
    dog_id = await register_file(db, "C:/docs/dog.md")
    cat_id = await register_file(db, "C:/docs/cat.md")
    await service.upsert_chunks(
        dog_id, "C:/docs/dog.md", [Chunk(seq=0, text="Пушистый пёс охраняет дом", locator="стр. 1")]
    )
    await service.upsert_chunks(
        cat_id,
        "C:/docs/cat.md",
        [Chunk(seq=0, text="Кошка спит на подоконнике весь день", locator="")],
    )

    found = await service.search("собака", k=5)
    assert any("пёс" in p.text for p in found)  # нашлось семантикой, не словами

    found_lexical = await service.search("кошка на подоконнике", k=5)
    assert any("Кошка" in p.text for p in found_lexical)


async def test_reindex_replaces_old_chunks(db: Database) -> None:
    service = make_service(db, FakeEmbedder(axes=["бюджет"]))
    file_id = await register_file(db, "C:/docs/plan.md")
    await service.upsert_chunks(
        file_id, "C:/docs/plan.md", [Chunk(seq=0, text="Старый бюджет сто тысяч", locator="")]
    )
    await service.upsert_chunks(
        file_id, "C:/docs/plan.md", [Chunk(seq=0, text="Новый бюджет двести тысяч", locator="")]
    )
    found = await service.search("какой бюджет?", k=10)
    texts = [p.text for p in found]
    assert any("Новый" in t for t in texts)
    assert not any("Старый" in t for t in texts)
    assert await service.chunk_count() == 1


async def test_delete_doc_removes_from_search(db: Database) -> None:
    service = make_service(db, FakeEmbedder(axes=["ладог"]))
    file_id = await register_file(db, "C:/docs/trip.md")
    await service.upsert_chunks(
        file_id,
        "C:/docs/trip.md",
        [Chunk(seq=0, text="Маршрут по Ладоге на байдарках", locator="")],
    )
    assert await service.search("маршрут по Ладоге", k=5)
    await service.delete_doc(file_id)
    assert await service.search("маршрут по Ладоге", k=5) == []
    assert await service.chunk_count() == 0


async def test_search_degrades_to_lexical_without_embeddings(db: Database) -> None:
    good = FakeEmbedder(axes=["смет"])
    service = make_service(db, good)
    file_id = await register_file(db, "C:/docs/smeta.md")
    await service.upsert_chunks(
        file_id, "C:/docs/smeta.md", [Chunk(seq=0, text="Смета на ремонт крыльца", locator="")]
    )
    # эмбеддер «упал» — поиск продолжает работать лексически
    degraded = make_service(db, BrokenEmbedder())
    found = await degraded.search("смета на ремонт", k=5)
    assert found and "Смета" in found[0].text


async def test_upsert_fails_atomically_when_embedder_down(db: Database) -> None:
    """Эмбеддер недоступен → старый индекс не разрушен, LLMError уходит наверх."""
    service = make_service(db, FakeEmbedder(axes=["крыльц"]))
    file_id = await register_file(db, "C:/docs/smeta.md")
    await service.upsert_chunks(
        file_id, "C:/docs/smeta.md", [Chunk(seq=0, text="Смета на ремонт крыльца", locator="")]
    )
    broken = make_service(db, BrokenEmbedder())
    with pytest.raises(LLMError):
        await broken.upsert_chunks(
            file_id, "C:/docs/smeta.md", [Chunk(seq=0, text="Новая версия сметы", locator="")]
        )
    assert await service.chunk_count() == 1  # старый чанк на месте


async def test_search_documents_tool_formats_citations(db: Database) -> None:
    service = make_service(db, FakeEmbedder(axes=["договор"]))
    file_id = await register_file(db, "C:/docs/contract.pdf")
    await service.upsert_chunks(
        file_id,
        "C:/docs/contract.pdf",
        [Chunk(seq=0, text="Договор подряда с ООО Ромашка на сумму 100000", locator="стр. 3")],
    )
    (tool,) = build_rag_tools(service, top_k=5, snippet_chars=100, configured=True)
    assert tool.name == "search_documents"
    result = await tool.handler(tool.args_schema(query="договор с Ромашкой"))
    assert "contract.pdf" in result
    assert "стр. 3" in result
    assert "Ромашка" in result


async def test_search_documents_tool_honest_when_empty(db: Database) -> None:
    service = make_service(db, FakeEmbedder(axes=[]))
    (tool,) = build_rag_tools(service, top_k=5, snippet_chars=100, configured=True)
    result = await tool.handler(tool.args_schema(query="несуществующая тема"))
    assert "не найдено" in result


async def test_search_documents_tool_hints_when_not_configured(db: Database) -> None:
    service = make_service(db, FakeEmbedder(axes=[]))
    (tool,) = build_rag_tools(service, top_k=5, snippet_chars=100, configured=False)
    result = await tool.handler(tool.args_schema(query="что угодно"))
    assert "не настроен" in result

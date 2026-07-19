"""Сквозной сценарий Sprint 4 (DoD): папка с документами → индексация →
вопрос по содержимому → ответ инструмента с файлом и местом."""

from pathlib import Path

import pytest

from sba.infra.config import RagConfig
from sba.infra.db import Database
from sba.infra.vectors import VectorStore
from sba.modules.indexer.service import IndexerService
from sba.modules.indexer.store import CatalogStore
from sba.modules.rag.service import RAGService
from sba.modules.rag.store import ChunkStore
from sba.modules.rag.tools import build_rag_tools


class KeywordEmbedder:
    """Оси-ключевые слова: детерминированная «семантика» без модели."""

    AXES = ("ладог", "бюджет", "договор")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0 if axis in text.lower() else 0.0 for axis in self.AXES] + [0.1]
            for text in texts
        ]


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


async def test_folder_to_cited_answer(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "expedition.md").write_text(
        "# Экспедиция\n\n## Маршрут\nИдём по Ладоге на байдарках две недели.\n"
        "## Бюджет\nОбщий бюджет похода — сто тысяч рублей.\n",
        encoding="utf-8",
    )
    (source / "misc.txt").write_text("Список покупок: хлеб, молоко, батарейки.", "utf-8")

    vectors = VectorStore.in_memory()
    rag = RAGService(ChunkStore(db), vectors, KeywordEmbedder(), embed_batch=2)
    config = RagConfig(sources=[source], dialog_cooldown_seconds=0.0)
    indexer = IndexerService(CatalogStore(db), rag, config)

    await indexer.scan_once()
    assert await indexer.process_queue() == 2

    (tool,) = build_rag_tools(rag, top_k=5, snippet_chars=200, configured=True)
    answer = await tool.handler(tool.args_schema(query="какой бюджет экспедиции?"))
    assert "expedition.md" in answer
    assert "раздел «Бюджет»" in answer
    assert "сто тысяч" in answer

    # новый файл в папке находится после следующего скана (DoD «в течение минут»)
    (source / "contract.txt").write_text("Договор аренды байдарок подписан.", "utf-8")
    await indexer.scan_once()
    await indexer.process_queue()
    answer2 = await tool.handler(tool.args_schema(query="что с договором аренды?"))
    assert "contract.txt" in answer2

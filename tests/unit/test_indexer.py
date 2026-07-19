"""Индексатор: скан-диффы, очередь с повторами, обработка удалений."""

from pathlib import Path

import pytest

from sba.infra.config import RagConfig
from sba.infra.db import Database
from sba.llm.gateway import LLMError
from sba.modules.indexer.service import IndexerService
from sba.modules.indexer.store import CatalogStore
from sba.modules.rag.interface import Chunk


class FakeIndex:
    """Запоминает вызовы вместо настоящего RAGService."""

    def __init__(self) -> None:
        self.upserts: list[tuple[str, str, int]] = []
        self.deletes: list[str] = []
        self.fail_with: Exception | None = None

    async def upsert_chunks(self, file_id: str, path: str, chunks: list[Chunk]) -> int:
        if self.fail_with is not None:
            raise self.fail_with
        self.upserts.append((file_id, path, len(chunks)))
        return len(chunks)

    async def delete_doc(self, file_id: str) -> None:
        self.deletes.append(file_id)


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_indexer(
    db: Database, source: Path, index: FakeIndex
) -> tuple[IndexerService, CatalogStore]:
    catalog = CatalogStore(db)
    config = RagConfig(sources=[source], dialog_cooldown_seconds=0.0)
    return IndexerService(catalog, index, config), catalog


async def test_new_file_scanned_and_indexed(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "note.md").write_text("# План\nМаршрут по Ладоге, бюджет сто тысяч.", "utf-8")
    index = FakeIndex()
    indexer, catalog = make_indexer(db, source, index)

    enqueued, deleted = await indexer.scan_once()
    assert (enqueued, deleted) == (1, 0)
    assert await indexer.process_queue() == 1
    assert len(index.upserts) == 1
    record = await catalog.get(str(source / "note.md"))
    assert record is not None and record.status == "indexed"
    assert await catalog.queue_size() == 0


async def test_unchanged_file_not_reindexed(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "note.md").write_text("Просто заметка о встрече с Иваном.", "utf-8")
    index = FakeIndex()
    indexer, _ = make_indexer(db, source, index)
    await indexer.scan_once()
    await indexer.process_queue()

    enqueued, _ = await indexer.scan_once()
    assert enqueued == 0
    await indexer.process_queue()
    assert len(index.upserts) == 1  # повторной индексации не было


async def test_changed_file_reindexed(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    file = source / "note.md"
    file.write_text("Первая версия заметки о проекте.", "utf-8")
    index = FakeIndex()
    indexer, _ = make_indexer(db, source, index)
    await indexer.scan_once()
    await indexer.process_queue()

    file.write_text("Вторая версия заметки о проекте, с правками.", "utf-8")
    enqueued, _ = await indexer.scan_once()
    assert enqueued == 1
    await indexer.process_queue()
    assert len(index.upserts) == 2


async def test_deleted_file_removed_from_index(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    file = source / "note.md"
    file.write_text("Заметка, которую скоро удалят навсегда.", "utf-8")
    index = FakeIndex()
    indexer, catalog = make_indexer(db, source, index)
    await indexer.scan_once()
    await indexer.process_queue()
    record = await catalog.get(str(file))
    assert record is not None

    file.unlink()
    _, deleted = await indexer.scan_once()
    assert deleted == 1
    await indexer.process_queue()
    assert index.deletes == [record.id]
    assert await catalog.get(str(file)) is None


async def test_llm_error_keeps_file_in_queue(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "note.md").write_text("Заметка при недоступной модели эмбеддинга.", "utf-8")
    index = FakeIndex()
    index.fail_with = LLMError("Ollama выключена")
    indexer, catalog = make_indexer(db, source, index)
    await indexer.scan_once()
    assert await indexer.process_queue() == 0
    assert await catalog.queue_size() == 1  # попытки не сожжены, файл ждёт

    index.fail_with = None
    assert await indexer.process_queue() == 1
    assert await catalog.queue_size() == 0


async def test_broken_file_marked_failed_after_retries(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    file = source / "broken.pdf"
    file.write_bytes(b"this is not a pdf at all")
    index = FakeIndex()
    indexer, catalog = make_indexer(db, source, index)
    await indexer.scan_once()

    # каждый вызов process_queue сжигает одну попытку (без горячего цикла)
    for _ in range(3):
        await indexer.process_queue()
    record = await catalog.get(str(file))
    assert record is not None and record.status == "failed"
    assert record.error
    assert await catalog.queue_size() == 0
    assert index.upserts == []


async def test_hidden_dirs_and_foreign_extensions_skipped(
    db: Database, tmp_path: Path
) -> None:
    source = tmp_path / "docs"
    (source / ".obsidian").mkdir(parents=True)
    (source / ".obsidian" / "config.md").write_text("служебное, не индексировать", "utf-8")
    (source / "photo.jpg").write_bytes(b"\xff\xd8")
    (source / "real.txt").write_text("Настоящий документ для индексации.", "utf-8")
    index = FakeIndex()
    indexer, _ = make_indexer(db, source, index)
    enqueued, _ = await indexer.scan_once()
    assert enqueued == 1
    await indexer.process_queue()
    assert [u[1] for u in index.upserts] == [str(source / "real.txt")]


async def test_stats_text_reports_state(db: Database, tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "a.txt").write_text("Документ номер один, про сметы.", "utf-8")
    index = FakeIndex()
    indexer, _ = make_indexer(db, source, index)
    await indexer.scan_once()
    await indexer.process_queue()
    text = await indexer.stats_text()
    assert "проиндексировано файлов: 1" in text
    assert "в очереди: 0" in text

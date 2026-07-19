"""Ручная переиндексация документов.

ВАЖНО: запускать только при ОСТАНОВЛЕННОМ ассистенте — embedded-хранилище
Qdrant нельзя открыть из двух процессов одновременно.

Windows (PowerShell, из папки проекта):
    .\\.venv\\Scripts\\Activate.ps1
    python scripts\\reindex.py            # доиндексировать новое/изменённое
    python scripts\\reindex.py --full     # снести индекс и построить заново
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from sba.infra.config import load_config
from sba.infra.db import Database
from sba.infra.logging import setup_logging
from sba.infra.vectors import VectorStore
from sba.llm.config import load_models_config
from sba.llm.service import ModelGateway
from sba.modules.indexer.service import IndexerService
from sba.modules.indexer.store import CatalogStore
from sba.modules.rag.service import DOCUMENTS_COLLECTION, RAGService
from sba.modules.rag.store import ChunkStore

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


async def wipe(db: Database, vectors: VectorStore) -> None:
    for table in ("rag_queue", "rag_chunks_fts", "rag_chunks", "rag_files"):
        await db.execute(f"DELETE FROM {table}")
    await vectors.drop_collection(DOCUMENTS_COLLECTION)
    print("Старый индекс удалён.")


async def main(full: bool) -> None:
    config = load_config(CONFIG_DIR)
    setup_logging("WARNING", "console")  # прогресс печатаем сами, без шума логов
    if not config.modules.rag.sources:
        print("Список папок modules.rag.sources пуст — добавьте его в config/local.yaml.")
        return

    db = await Database.open(config.app.data_dir / "sba.db")
    vectors = VectorStore.open(config.app.data_dir / "qdrant")
    gateway = ModelGateway(load_models_config(CONFIG_DIR / "models.yaml"))
    rag = RAGService(
        ChunkStore(db), vectors, gateway, embed_batch=config.modules.rag.embed_batch
    )
    indexer = IndexerService(CatalogStore(db), rag, config.modules.rag)
    try:
        if full:
            await wipe(db, vectors)
        enqueued, deleted = await indexer.scan_once()
        print(f"Скан завершён: в очереди {enqueued} файлов, удалено {deleted}.")
        processed = await indexer.process_queue(respect_cooldown=False)
        chunks = await rag.chunk_count()
        print(f"Проиндексировано файлов: {processed}; фрагментов в индексе: {chunks}.")
        print(await indexer.stats_text())
    finally:
        await gateway.aclose()
        await vectors.close()
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Переиндексация документов")
    parser.add_argument(
        "--full", action="store_true", help="снести индекс и построить заново"
    )
    asyncio.run(main(parser.parse_args().full))

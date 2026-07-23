"""Прямой поиск по индексу — мимо LLM. Показывает, что реально находит
RAGService, чтобы отделить качество индекса/поиска от капризов модели.

ВАЖНО: запускать при ОСТАНОВЛЕННОМ ассистенте — embedded-хранилище Qdrant
нельзя открыть из двух процессов одновременно.

Windows (PowerShell, из папки проекта):
    .\\.venv\\Scripts\\Activate.ps1
    python scripts\\search_probe.py                       # два тестовых запроса
    python scripts\\search_probe.py "свой запрос"         # произвольный запрос
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
from sba.modules.rag.service import RAGService
from sba.modules.rag.store import ChunkStore

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_QUERIES = ["тег сбапроверка", "квартальныйбюджетсба2026"]


async def main(queries: list[str], k: int) -> None:
    config = load_config(CONFIG_DIR)
    setup_logging("WARNING", "console")
    db = await Database.open(config.app.data_dir / "sba.db")
    vectors = VectorStore.open(config.app.data_dir / "qdrant")
    gateway = ModelGateway(load_models_config(CONFIG_DIR / "models.yaml"))
    rag = RAGService(
        ChunkStore(db), vectors, gateway, embed_batch=config.modules.rag.embed_batch
    )
    try:
        total = await rag.chunk_count()
        print(f"Фрагментов в индексе всего: {total}\n")
        for query in queries:
            print(f"══ Запрос: «{query}» ══")
            passages = await rag.search(query, k)
            if not passages:
                print("  (ничего не найдено)\n")
                continue
            for i, p in enumerate(passages, start=1):
                name = Path(p.path).name
                place = f" [{p.locator}]" if p.locator else ""
                snippet = " ".join(p.text.split())[:160]
                print(f"  {i}. {name}{place}")
                print(f"     {snippet}")
            print()
    finally:
        await gateway.aclose()
        await vectors.close()
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Прямой поиск по индексу (мимо LLM)")
    parser.add_argument("queries", nargs="*", help="запросы (по умолчанию — тестовые)")
    parser.add_argument("-k", type=int, default=5, help="сколько результатов на запрос")
    args = parser.parse_args()
    asyncio.run(main(args.queries or DEFAULT_QUERIES, args.k))

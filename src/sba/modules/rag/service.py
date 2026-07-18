"""RAG Service: гибридный поиск по документам с цитатами.

Гибрид = dense (Qdrant, эмбеддинги bge-m3) + лексический (SQLite FTS5
с префиксами под русскую морфологию), слияние RRF. Отступление от плана
(нативный sparse Qdrant) задокументировано в docs/06-roadmap.md: sparse
потребовал бы отдельный энкодер, FTS5 уже обкатан на памяти и работает ру/en.

При недоступной эмбеддинг-модели поиск деградирует до лексического —
ассистент продолжает работать (урок Sprint 2: всё внешнее может отказать).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from sba.infra.ranking import rrf_merge
from sba.infra.vectors import VectorPoint, VectorStore
from sba.llm.gateway import Embedder, LLMError
from sba.modules.rag.interface import Chunk
from sba.modules.rag.store import ChunkStore

log = structlog.get_logger(__name__)

DOCUMENTS_COLLECTION = "documents"
# кандидатов от каждой половины гибрида берём с запасом до слияния
CANDIDATE_POOL = 20


@dataclass(frozen=True)
class Passage:
    path: str
    locator: str
    text: str


class RAGService:
    """Реализует rag.interface.DocumentIndex (для индексатора) + поиск (для инструмента)."""

    def __init__(
        self,
        chunks: ChunkStore,
        vectors: VectorStore,
        embedder: Embedder,
        embed_batch: int = 8,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        self._embed_batch = embed_batch

    # ── индексация (DocumentIndex) ───────────────────────────────────────────

    async def upsert_chunks(self, file_id: str, path: str, chunks: list[Chunk]) -> int:
        """Заменяет содержимое файла в индексе; возвращает число чанков.

        Эмбеддинги считаются ДО замены в SQLite: если модель недоступна,
        старый индекс остаётся целым, файл останется в очереди на повтор.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self._embed_batch):
            batch = chunks[start : start + self._embed_batch]
            vectors.extend(await self._embedder.embed([c.text for c in batch]))
        if vectors:
            await self._vectors.ensure_collection(DOCUMENTS_COLLECTION, len(vectors[0]))

        old_ids = await self._chunks.ids_for_file(file_id)
        await self._vectors.delete(DOCUMENTS_COLLECTION, old_ids)
        stored = await self._chunks.replace_for_file(file_id, chunks)
        await self._vectors.upsert(
            DOCUMENTS_COLLECTION,
            [
                VectorPoint(
                    id=record.id,
                    vector=vector,
                    payload={"file_id": file_id, "path": path},
                )
                for record, vector in zip(stored, vectors, strict=True)
            ],
        )
        return len(stored)

    async def delete_doc(self, file_id: str) -> None:
        ids = await self._chunks.delete_for_file(file_id)
        await self._vectors.delete(DOCUMENTS_COLLECTION, ids)

    # ── поиск ────────────────────────────────────────────────────────────────

    async def search(self, query: str, k: int) -> list[Passage]:
        lexical = await self._chunks.lexical_search(query, CANDIDATE_POOL)
        dense_ids: list[str] = []
        try:
            query_vector = (await self._embedder.embed([query]))[0]
            hits = await self._vectors.search(
                DOCUMENTS_COLLECTION, query_vector, CANDIDATE_POOL
            )
            dense_ids = [hit.id for hit in hits]
        except LLMError as exc:
            # деградация до лексического поиска — лучше, чем отказ
            log.warning("rag_dense_search_unavailable", error=str(exc))

        merged = rrf_merge([[c.id for c in lexical], dense_ids], limit=k)
        by_id = {c.id: c for c in lexical}
        missing = [chunk_id for chunk_id in merged if chunk_id not in by_id]
        by_id.update(await self._chunks.by_ids(missing))
        return [
            Passage(path=by_id[i].path, locator=by_id[i].locator, text=by_id[i].text)
            for i in merged
            if i in by_id
        ]

    async def chunk_count(self) -> int:
        return await self._chunks.count()

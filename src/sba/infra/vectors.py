"""Обёртка над Qdrant в embedded-режиме (ADR-3): файлы на диске, без сервера.

Клиент Qdrant синхронный — все вызовы уходят в worker-поток через
asyncio.to_thread и сериализуются одним замком (local-режим не рассчитан
на конкурентный доступ). Один экземпляр на процесс: embedded-хранилище
нельзя открыть дважды.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import QdrantClient, models

log = structlog.get_logger(__name__)


class VectorStoreError(Exception):
    pass


@dataclass(frozen=True)
class VectorPoint:
    id: str  # канонический UUID (требование Qdrant к id точек)
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._lock = asyncio.Lock()

    @classmethod
    def open(cls, path: Path) -> VectorStore:
        path.mkdir(parents=True, exist_ok=True)
        return cls(QdrantClient(path=str(path)))

    @classmethod
    def in_memory(cls) -> VectorStore:
        """Для тестов: то же API, состояние в RAM."""
        return cls(QdrantClient(location=":memory:"))

    async def ensure_collection(self, name: str, dim: int) -> None:
        """Создаёт коллекцию при первом обращении; проверяет размерность.

        Размерность известна только после первого эмбеддинга, поэтому
        создание ленивое. Несовпадение = сменили эмбеддинг-модель без
        переиндексации — честно отказываем с подсказкой.
        """
        async with self._lock:
            await asyncio.to_thread(self._ensure_collection_sync, name, dim)

    def _ensure_collection_sync(self, name: str, dim: int) -> None:
        if not self._client.collection_exists(name):
            self._client.create_collection(
                name,
                vectors_config=models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                ),
            )
            log.info("vector_collection_created", name=name, dim=dim)
            return
        info = self._client.get_collection(name)
        params = info.config.params.vectors
        existing = params.size if isinstance(params, models.VectorParams) else None
        if existing != dim:
            raise VectorStoreError(
                f"коллекция {name!r} создана под размерность {existing}, "
                f"а эмбеддинг-модель выдаёт {dim}. Смените модель обратно "
                f"или выполните полную переиндексацию (scripts/reindex.py)."
            )

    async def upsert(self, name: str, points: list[VectorPoint]) -> None:
        if not points:
            return
        structs = [
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        async with self._lock:
            await asyncio.to_thread(self._client.upsert, name, structs)

    async def search(
        self,
        name: str,
        vector: list[float],
        k: int,
        payload_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        query_filter = None
        if payload_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in payload_filter.items()
                ]
            )
        async with self._lock:
            if not await asyncio.to_thread(self._client.collection_exists, name):
                return []
            response = await asyncio.to_thread(
                self._client.query_points,
                name,
                query=vector,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )
        return [
            VectorHit(id=str(p.id), score=p.score, payload=p.payload or {})
            for p in response.points
        ]

    async def delete(self, name: str, ids: list[str]) -> None:
        if not ids:
            return
        async with self._lock:
            if not await asyncio.to_thread(self._client.collection_exists, name):
                return
            await asyncio.to_thread(
                self._client.delete,
                name,
                points_selector=models.PointIdsList(points=list(ids)),
            )

    async def drop_collection(self, name: str) -> None:
        """Для полной переиндексации (scripts/reindex.py)."""
        async with self._lock:
            if await asyncio.to_thread(self._client.collection_exists, name):
                await asyncio.to_thread(self._client.delete_collection, name)
                log.info("vector_collection_dropped", name=name)

    async def count(self, name: str) -> int:
        async with self._lock:
            if not await asyncio.to_thread(self._client.collection_exists, name):
                return 0
            result = await asyncio.to_thread(self._client.count, name)
        return int(result.count)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._client.close)

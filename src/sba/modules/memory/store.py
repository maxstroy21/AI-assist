"""Хранилище памяти: SQLite + FTS5 + векторный recall (Sprint 4).

Факты не удаляются физически: «забудь» ставит retracted_at, вытеснение —
superseded_by (история решений сохраняется, docs/02-architecture.md §3).
Поиск гибридный: FTS5 с префиксами слов под русскую морфологию
(«Ивана» → «иван*») + семантическая близость (Qdrant-коллекция memory),
слияние RRF. Векторная часть строго best-effort: при недоступной
эмбеддинг-модели память продолжает работать на FTS — «запомни/вспомни»
не должны зависеть от Ollama.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog

from sba.infra.db import Database
from sba.infra.ranking import rrf_merge
from sba.infra.text import build_fts_query
from sba.infra.vectors import VectorPoint, VectorStore
from sba.llm.gateway import Embedder

log = structlog.get_logger(__name__)

ACTIVE = "superseded_by IS NULL AND retracted_at IS NULL"

MEMORY_COLLECTION = "memory"
VECTOR_POOL = 12  # кандидатов от векторной половины до слияния


def _point_id(fact_id: str) -> str:
    """id факта (32 hex) → канонический UUID для Qdrant."""
    return str(uuid.UUID(hex=fact_id))


@dataclass(frozen=True)
class FactView:
    id: str
    type: str
    subject: str
    content: str
    created_at: str


def _view(row: object) -> FactView:
    return FactView(
        id=row["id"],  # type: ignore[index]
        type=row["type"],  # type: ignore[index]
        subject=row["subject"],  # type: ignore[index]
        content=row["content"],  # type: ignore[index]
        created_at=row["created_at"],  # type: ignore[index]
    )


class MemoryStore:
    def __init__(
        self,
        db: Database,
        vectors: VectorStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._db = db
        self._vectors = vectors
        self._embedder = embedder

    @property
    def _vector_enabled(self) -> bool:
        return self._vectors is not None and self._embedder is not None

    async def add(
        self,
        user_id: str,
        fact_type: str,
        subject: str,
        content: str,
        source: str = "explicit",
        confidence: float = 1.0,
    ) -> FactView:
        fact_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        if fact_type == "preference":
            # предпочтение по той же теме вытесняет старое (актуально одно)
            superseded = await self._db.fetch_all(
                f"SELECT id FROM memory_facts WHERE user_id=? AND type='preference'"
                f" AND lower(subject)=lower(?) AND {ACTIVE}",
                (user_id, subject),
            )
            await self._db.execute(
                f"UPDATE memory_facts SET superseded_by=?, updated_at=?"
                f" WHERE user_id=? AND type='preference'"
                f" AND lower(subject)=lower(?) AND {ACTIVE}",
                (fact_id, now, user_id, subject),
            )
            await self._drop_vectors([str(r["id"]) for r in superseded])
        await self._db.execute(
            "INSERT INTO memory_facts"
            " (id, user_id, type, subject, content, source, confidence,"
            "  created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (fact_id, user_id, fact_type, subject, content, source, confidence, now, now),
        )
        await self._db.execute(
            "INSERT INTO memory_fts (subject, content, fact_id) VALUES (?, ?, ?)",
            (subject, content, fact_id),
        )
        await self._add_vector(fact_id, user_id, subject, content)
        return FactView(
            id=fact_id, type=fact_type, subject=subject, content=content, created_at=now
        )

    async def search(self, user_id: str, query: str, k: int = 6) -> list[FactView]:
        fts = build_fts_query(query)
        if fts is None:
            # «что ты помнишь?» — показываем свежие факты
            rows = await self._db.fetch_all(
                f"SELECT * FROM memory_facts WHERE user_id=? AND {ACTIVE}"
                f" ORDER BY created_at DESC LIMIT ?",
                (user_id, k),
            )
            return [_view(r) for r in rows]
        rows = await self._db.fetch_all(
            f"SELECT f.* FROM memory_fts m JOIN memory_facts f ON f.id = m.fact_id"
            f" WHERE memory_fts MATCH ? AND f.user_id=? AND {ACTIVE}"
            f" ORDER BY rank LIMIT ?",
            (fts, user_id, k),
        )
        lexical = [_view(r) for r in rows]
        semantic_ids = await self._vector_search(user_id, query)
        if not semantic_ids:
            return lexical[:k]
        merged = rrf_merge([[f.id for f in lexical], semantic_ids], limit=k)
        by_id = {f.id: f for f in lexical}
        missing = [fact_id for fact_id in merged if fact_id not in by_id]
        by_id.update(await self._by_ids(user_id, missing))
        return [by_id[i] for i in merged if i in by_id]

    async def retract(self, user_id: str, query: str) -> list[FactView]:
        """Мягкое «забудь»: помечает найденные факты retracted_at."""
        matched = await self.search(user_id, query, k=20)
        if build_fts_query(query) is None:
            return []  # «забудь всё» без уточнения — не выполняем
        now = datetime.now(UTC).isoformat()
        for fact in matched:
            await self._db.execute(
                "UPDATE memory_facts SET retracted_at=?, updated_at=? WHERE id=?",
                (now, now, fact.id),
            )
        await self._drop_vectors([f.id for f in matched])
        return matched

    async def _by_ids(self, user_id: str, ids: list[str]) -> dict[str, FactView]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = await self._db.fetch_all(
            f"SELECT * FROM memory_facts WHERE user_id=? AND {ACTIVE}"
            f" AND id IN ({placeholders})",
            (user_id, *ids),
        )
        return {fact.id: fact for fact in (_view(r) for r in rows)}

    # ── векторная половина (best-effort) ─────────────────────────────────────

    async def _add_vector(self, fact_id: str, user_id: str, subject: str, content: str) -> None:
        if not self._vector_enabled:
            return
        assert self._vectors is not None and self._embedder is not None
        try:
            vector = (await self._embedder.embed([f"{subject}. {content}"]))[0]
            await self._vectors.ensure_collection(MEMORY_COLLECTION, len(vector))
            await self._vectors.upsert(
                MEMORY_COLLECTION,
                [
                    VectorPoint(
                        id=_point_id(fact_id),
                        vector=vector,
                        payload={"user_id": user_id, "fact_id": fact_id},
                    )
                ],
            )
        except Exception as exc:  # память работает и без векторов
            log.warning("memory_vector_add_failed", error=str(exc))

    async def _vector_search(self, user_id: str, query: str) -> list[str]:
        if not self._vector_enabled:
            return []
        assert self._vectors is not None and self._embedder is not None
        try:
            vector = (await self._embedder.embed([query]))[0]
            hits = await self._vectors.search(
                MEMORY_COLLECTION, vector, VECTOR_POOL, {"user_id": user_id}
            )
            return [str(hit.payload.get("fact_id", "")) for hit in hits]
        except Exception as exc:
            log.warning("memory_vector_search_failed", error=str(exc))
            return []

    async def _drop_vectors(self, fact_ids: list[str]) -> None:
        if not self._vector_enabled or not fact_ids:
            return
        assert self._vectors is not None
        try:
            await self._vectors.delete(
                MEMORY_COLLECTION, [_point_id(fact_id) for fact_id in fact_ids]
            )
        except Exception as exc:
            log.warning("memory_vector_delete_failed", error=str(exc))

    async def preferences(self, user_id: str) -> list[FactView]:
        rows = await self._db.fetch_all(
            f"SELECT * FROM memory_facts"
            f" WHERE user_id=? AND type='preference' AND {ACTIVE}"
            f" ORDER BY created_at",
            (user_id,),
        )
        return [_view(r) for r in rows]

    async def count_active(self, user_id: str) -> int:
        row = await self._db.fetch_one(
            f"SELECT COUNT(*) AS n FROM memory_facts WHERE user_id=? AND {ACTIVE}",
            (user_id,),
        )
        return int(row["n"]) if row else 0

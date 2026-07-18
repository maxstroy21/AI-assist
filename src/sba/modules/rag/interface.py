"""Публичный интерфейс модуля rag для других модулей (правило границ №1:
чужой модуль импортируется только через его interface.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    """Фрагмент документа, готовый к индексации."""

    seq: int
    text: str
    locator: str  # «стр. 3», «раздел …», «фрагмент 2» — для цитирования


class DocumentIndex(Protocol):
    """То, что индексатор может делать с поисковым индексом."""

    async def upsert_chunks(self, file_id: str, path: str, chunks: list[Chunk]) -> int: ...

    async def delete_doc(self, file_id: str) -> None: ...

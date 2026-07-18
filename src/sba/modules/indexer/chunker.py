"""Чанкер: блоки экстрактора → фрагменты ~500–800 токенов с перекрытием.

Бюджеты в символах (~4 симв./токен для русского), границы — по абзацам,
чтобы фрагмент оставался связным. Локатор наследуется от блока
(страница/раздел); безымянные блоки нумеруются «фрагмент N».
"""

from __future__ import annotations

from sba.modules.indexer.extractors import Block
from sba.modules.rag.interface import Chunk

MIN_CHUNK_CHARS = 20  # огрызки короче не несут смысла и шумят в поиске


def make_chunks(blocks: list[Block], chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        for piece in _split_text(block.text, chunk_chars, overlap_chars):
            if len(piece.strip()) < MIN_CHUNK_CHARS:
                continue
            locator = block.locator or f"фрагмент {len(chunks) + 1}"
            chunks.append(Chunk(seq=len(chunks), text=piece.strip(), locator=locator))
    return chunks


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        if len(paragraph) <= size:
            buffer = paragraph
        else:
            pieces.extend(_split_hard(paragraph, size, overlap))
            buffer = ""
    if buffer:
        pieces.append(buffer)
    return pieces


def _split_hard(text: str, size: int, overlap: int) -> list[str]:
    """Абзац длиннее бюджета: режем с перекрытием, край — по ближайшему пробелу."""
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space > start:
                end = space
        pieces.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces

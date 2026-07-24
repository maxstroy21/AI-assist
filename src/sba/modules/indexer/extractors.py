"""Извлечение текста из файлов: PDF (pypdf), DOCX (python-docx), XLSX
(openpyxl), MD, TXT.

Все функции синхронные и CPU-bound — вызывающий обязан уводить их
в worker-поток (asyncio.to_thread) и оборачивать таймаутом.
Отступление от плана: PyMuPDF заменён на чистый pypdf (без бинарных
зависимостей); OCR-fallback для сканов — Sprint 8 по плану.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class ExtractError(Exception):
    pass


@dataclass(frozen=True)
class Block:
    """Структурный блок текста с локатором для цитирования."""

    text: str
    locator: str  # «стр. 3», «раздел …»; пусто — чанкер пронумерует сам


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".md", ".txt"}

# Потолки для XLSX: таблица владельца может быть на десятки тысяч строк —
# «всё внешнее обязано иметь границу» (урок проекта). Чанкер потом дробит
# блок сам, но строить гигантскую строку в памяти мы не будем.
XLSX_MAX_ROWS_PER_SHEET = 5000
XLSX_MAX_COLS_PER_ROW = 64


def extract(path: Path) -> list[Block]:
    """Файл → блоки текста. ExtractError — файл прочитать не удалось."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix == ".xlsx":
            return _extract_xlsx(path)
        if suffix == ".md":
            return _extract_markdown(path)
        if suffix == ".txt":
            return [Block(text=_read_text(path), locator="")]
    except ExtractError:
        raise
    except Exception as exc:  # у каждой библиотеки свой зоопарк исключений
        raise ExtractError(f"{path.name}: {exc}") from exc
    raise ExtractError(f"{path.name}: расширение {suffix!r} не поддерживается")


def _read_text(path: Path) -> str:
    """UTF-8, при неудаче — cp1251 (типично для старых Windows-файлов)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1251", errors="replace")


def _extract_pdf(path: Path) -> list[Block]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ExtractError(f"{path.name}: PDF зашифрован")
    blocks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            blocks.append(Block(text=text, locator=f"стр. {page_no}"))
    return blocks


def _extract_docx(path: Path) -> list[Block]:
    import docx

    document = docx.Document(str(path))
    blocks: list[Block] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(Block(text="\n".join(buffer), locator=heading))
            buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "") if paragraph.style else ""
        if style.startswith(("Heading", "Заголовок", "Title")):
            flush()
            heading = f"раздел «{text[:60]}»"
            continue
        buffer.append(text)
    flush()
    return blocks


def _format_cell(value: object) -> str:
    """Ячейка → короткая строка для поиска. Формулы уже посчитаны
    (data_only), даты приводим к ISO без микросекунд."""
    if value is None:
        return ""
    if isinstance(value, bool):  # bool раньше int — иначе True станет «1»
        return "да" if value else "нет"
    if isinstance(value, datetime):
        text = value.isoformat(sep=" ")
        return text[:-3] if text.endswith(":00") else text  # секунды-нули — лишние
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # 100.0 → «100», чтобы искалось как в тексте
    return str(value).strip()


def _extract_xlsx(path: Path) -> list[Block]:
    """Каждый лист → блок; строки склеены ` | `, пустые пропущены.
    read_only — не держим весь файл в памяти; data_only — значения
    формул, а не их текст (Excel кэширует их при сохранении)."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        blocks: list[Block] = []
        for sheet in workbook.worksheets:
            lines: list[str] = []
            truncated = False
            for row in sheet.iter_rows(values_only=True):
                if len(lines) >= XLSX_MAX_ROWS_PER_SHEET:
                    truncated = True
                    break
                cells = [_format_cell(v) for v in row[:XLSX_MAX_COLS_PER_ROW]]
                while cells and not cells[-1]:  # хвостовые пустые ячейки — шум
                    cells.pop()
                if any(cells):
                    lines.append(" | ".join(cells))
            if truncated:
                lines.append(f"… (показаны первые {XLSX_MAX_ROWS_PER_SHEET} строк)")
            if lines:
                blocks.append(Block(text="\n".join(lines), locator=f"лист «{sheet.title}»"))
        return blocks
    finally:
        workbook.close()


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _extract_markdown(path: Path) -> list[Block]:
    text = _read_text(path)
    lines = text.splitlines()
    # YAML-frontmatter не несёт прозы — пропускаем
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                lines = lines[index + 1 :]
                break
    blocks: list[Block] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        joined = "\n".join(buffer).strip()
        if joined:
            blocks.append(Block(text=joined, locator=heading))
        buffer.clear()

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            heading = f"раздел «{match.group(2).strip()[:60]}»"
            continue
        buffer.append(line)
    flush()
    return blocks

"""Извлечение текста из файлов: PDF (pypdf), DOCX (python-docx), MD, TXT, XLSX.

Все функции синхронные и CPU-bound — вызывающий обязан уводить их
в worker-поток (asyncio.to_thread) и оборачивать таймаутом.
Отступление от плана: PyMuPDF заменён на чистый pypdf (без бинарных
зависимостей); OCR-fallback для сканов — Sprint 8, отдельным шагом.

XLSX читается через openpyxl (чистый Python) в потоковом режиме
(read_only) со значениями формул (data_only) — в поиск попадают числа
и подписи, а не текст формул. Markdown дополнительно обогащается тегами
и wikilinks Obsidian (детерминированно, без LLM), чтобы заметки искались
с учётом тегов.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path


class ExtractError(Exception):
    pass


@dataclass(frozen=True)
class Block:
    """Структурный блок текста с локатором для цитирования."""

    text: str
    locator: str  # «стр. 3», «раздел …», «лист …»; пусто — чанкер пронумерует сам


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".xlsx", ".xlsm"}

# Потолки-предохранители для XLSX (урок «всё внешнее обязано иметь потолок»):
# гигантская таблица не должна съесть память/время даже под общим таймаутом.
XLSX_MAX_ROWS_PER_SHEET = 5000
XLSX_MAX_CELLS = 100_000
XLSX_CELL_SEP = " │ "


def extract(path: Path) -> list[Block]:
    """Файл → блоки текста. ExtractError — файл прочитать не удалось."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix == ".docx":
            return _extract_docx(path)
        if suffix == ".md":
            return _extract_markdown(path)
        if suffix in (".xlsx", ".xlsm"):
            return _extract_xlsx(path)
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


# ── XLSX ─────────────────────────────────────────────────────────────────────


def _format_cell(value: object) -> str:
    """Ячейка → компактная строка, удобная для поиска."""
    if isinstance(value, dt.datetime):
        # полночь → только дата (частый случай дат без времени)
        if value.time() == dt.time(0, 0):
            return value.date().isoformat()
        return value.isoformat(sep=" ")
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # 100.0 → «100», иначе поиск «100» не сматчит
    if isinstance(value, bool):
        return "да" if value else "нет"
    return str(value)


def _extract_xlsx(path: Path) -> list[Block]:
    """Каждый лист → один блок; строки — ячейки через разделитель.

    read_only — потоковое чтение без загрузки всей книги в память;
    data_only — вычисленные значения вместо текста формул. Оговорка:
    для формул значение вычисляет Excel при сохранении; если книга ни разу
    не открывалась в Excel, у таких ячеек кэша нет и они придут пустыми.
    """
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        blocks: list[Block] = []
        cells_seen = 0
        for sheet in workbook.worksheets:
            lines: list[str] = []
            truncated = False
            for row_no, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                if row_no > XLSX_MAX_ROWS_PER_SHEET or cells_seen >= XLSX_MAX_CELLS:
                    truncated = True
                    break
                rendered = [
                    _format_cell(value) for value in row if value not in (None, "")
                ]
                cells_seen += len(row)
                if rendered:
                    lines.append(XLSX_CELL_SEP.join(rendered))
            if truncated:
                lines.append("… (лист обрезан по лимиту строк)")
            if lines:
                blocks.append(Block(text="\n".join(lines), locator=f"лист «{sheet.title}»"))
        return blocks
    finally:
        workbook.close()


# ── Markdown + обогащение Obsidian ───────────────────────────────────────────

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# инлайновый тег Obsidian: #слово без пробела после решётки (в отличие от
# заголовка «# Текст», где после решётки пробел). Латиница, кириллица, /, -, _.
INLINE_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z0-9_/Ѐ-ӿ-]+)")
# wikilink: [[Цель]] или [[Цель|Алиас]] (возможен якорь [[Цель#Раздел]])
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _split_frontmatter(lines: list[str]) -> tuple[list[str], list[str]]:
    """(строки frontmatter, остальные строки). Frontmatter — блок между --- ."""
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return lines[1:index], lines[index + 1 :]
    return [], lines


def _frontmatter_tags(front: list[str]) -> list[str]:
    """Теги из YAML-frontmatter: `tags: [a, b]`, YAML-список или `tags: a b`."""
    tags: list[str] = []
    index = 0
    while index < len(front):
        line = front[index]
        match = re.match(r"^\s*tags\s*:\s*(.*)$", line, re.IGNORECASE)
        if not match:
            index += 1
            continue
        rest = match.group(1).strip()
        if rest:
            rest = rest.strip("[]")
            for part in re.split(r"[,\s]+", rest):
                cleaned = part.strip().lstrip("#").strip("\"'")
                if cleaned:
                    tags.append(cleaned)
        else:
            # YAML-список на следующих строках: «  - тег»
            index += 1
            while index < len(front):
                item = re.match(r"^\s*-\s*(.+)$", front[index])
                if not item:
                    break
                cleaned = item.group(1).strip().lstrip("#").strip("\"'")
                if cleaned:
                    tags.append(cleaned)
                index += 1
            continue
        index += 1
    return tags


def _dedup(items: list[str]) -> list[str]:
    """Порядок сохраняем, регистронезависимый дедуп."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _extract_markdown(path: Path) -> list[Block]:
    text = _read_text(path)
    lines = text.splitlines()
    front, body_lines = _split_frontmatter(lines)
    body = "\n".join(body_lines)

    # обогащение Obsidian: теги (frontmatter + инлайн) и wikilinks
    tags = _frontmatter_tags(front) + INLINE_TAG_RE.findall(body)
    tags = _dedup(tags)
    links = _dedup(
        [raw.split("|", 1)[0].split("#", 1)[0].strip() for raw in WIKILINK_RE.findall(body)]
    )

    meta: list[Block] = []
    if tags:
        meta.append(Block(text="Теги: " + " ".join(f"#{t}" for t in tags), locator="теги"))
    if links:
        meta.append(Block(text="Связи: " + ", ".join(links), locator="связи"))

    blocks: list[Block] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        joined = "\n".join(buffer).strip()
        if joined:
            blocks.append(Block(text=joined, locator=heading))
        buffer.clear()

    for line in body_lines:
        match = HEADING_RE.match(line)
        if match:
            flush()
            heading = f"раздел «{match.group(2).strip()[:60]}»"
            continue
        buffer.append(line)
    flush()
    return meta + blocks

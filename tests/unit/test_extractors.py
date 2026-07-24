from datetime import datetime
from pathlib import Path

import docx as docx_lib
import pytest
from openpyxl import Workbook

from sba.modules.indexer.extractors import (
    XLSX_MAX_ROWS_PER_SHEET,
    ExtractError,
    extract,
)


def build_minimal_pdf(text: str) -> bytes:
    """Минимальный однострочный PDF, собранный вручную (без зависимостей)."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"",  # поток содержимого — ниже
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects[3] = (
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_txt_utf8(tmp_path: Path) -> None:
    file = tmp_path / "note.txt"
    file.write_text("Привет, документ", encoding="utf-8")
    blocks = extract(file)
    assert blocks[0].text == "Привет, документ"


def test_txt_cp1251_fallback(tmp_path: Path) -> None:
    file = tmp_path / "old.txt"
    file.write_bytes("Старый виндовый файл".encode("cp1251"))
    assert "Старый виндовый файл" in extract(file)[0].text


def test_markdown_headings_and_frontmatter(tmp_path: Path) -> None:
    file = tmp_path / "note.md"
    file.write_text(
        "---\ntags: [test]\n---\n"
        "Вступление без заголовка.\n"
        "# План экспедиции\nМаршрут по Ладоге.\n"
        "## Бюджет\nСто тысяч рублей.\n",
        encoding="utf-8",
    )
    blocks = extract(file)
    assert [b.locator for b in blocks] == [
        "",
        "раздел «План экспедиции»",
        "раздел «Бюджет»",
    ]
    assert "tags" not in blocks[0].text  # frontmatter пропущен
    assert "Ладоге" in blocks[1].text


def test_pdf_pages(tmp_path: Path) -> None:
    file = tmp_path / "doc.pdf"
    file.write_bytes(build_minimal_pdf("Expedition budget 100k"))
    blocks = extract(file)
    assert blocks[0].locator == "стр. 1"
    assert "Expedition budget" in blocks[0].text


def test_docx_headings(tmp_path: Path) -> None:
    document = docx_lib.Document()
    document.add_heading("Договор подряда", level=1)
    document.add_paragraph("Исполнитель обязуется выполнить работы.")
    document.add_paragraph("Срок — до конца года.")
    file = tmp_path / "contract.docx"
    document.save(str(file))
    blocks = extract(file)
    assert blocks[0].locator == "раздел «Договор подряда»"
    assert "Исполнитель" in blocks[0].text


def test_xlsx_sheets_and_values(tmp_path: Path) -> None:
    workbook = Workbook()
    first = workbook.active
    first.title = "Бюджет"
    first.append(["Статья", "Сумма", "Оплачено"])
    first.append(["Аренда лодки", 100000.0, True])
    first.append([None, None, None])  # пустая строка — пропускается
    first.append(["Провизия", 25000, False])
    second = workbook.create_sheet("Сроки")
    second.append(["Выезд", datetime(2026, 8, 1, 9, 0)])
    file = tmp_path / "expedition.xlsx"
    workbook.save(str(file))

    blocks = extract(file)

    assert [b.locator for b in blocks] == ["лист «Бюджет»", "лист «Сроки»"]
    budget = blocks[0].text
    assert "Статья | Сумма | Оплачено" in budget
    assert "Аренда лодки | 100000 | да" in budget  # float→int, bool→«да»
    assert "Провизия | 25000 | нет" in budget
    assert "\n\n" not in budget  # пустая строка не породила пустую линию
    assert "Выезд | 2026-08-01 09:00" in blocks[1].text


def test_xlsx_empty_sheet_produces_no_block(tmp_path: Path) -> None:
    workbook = Workbook()
    workbook.active.title = "Пусто"  # ни одной ячейки со значением
    file = tmp_path / "empty.xlsx"
    workbook.save(str(file))
    assert extract(file) == []


def test_xlsx_row_cap(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    for i in range(XLSX_MAX_ROWS_PER_SHEET + 50):
        sheet.append([f"строка {i}"])
    file = tmp_path / "big.xlsx"
    workbook.save(str(file))
    lines = extract(file)[0].text.splitlines()
    # ровно потолок строк данных + одна строка-пометка об усечении
    assert len(lines) == XLSX_MAX_ROWS_PER_SHEET + 1
    assert lines[-1].startswith("…")


def test_broken_xlsx_raises_extract_error(tmp_path: Path) -> None:
    file = tmp_path / "broken.xlsx"
    file.write_bytes(b"PK\x03\x04 not really a spreadsheet")
    with pytest.raises(ExtractError):
        extract(file)


def test_unsupported_extension(tmp_path: Path) -> None:
    file = tmp_path / "img.png"
    file.write_bytes(b"\x89PNG")
    with pytest.raises(ExtractError):
        extract(file)


def test_broken_pdf_raises_extract_error(tmp_path: Path) -> None:
    file = tmp_path / "broken.pdf"
    file.write_bytes("это не PDF вовсе".encode())
    with pytest.raises(ExtractError):
        extract(file)

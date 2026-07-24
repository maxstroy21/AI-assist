"""Живая проверка Sprint 8 (шаг 2): кладёт в индексируемую папку тестовую
таблицу Excel и заметку Obsidian с тегами и wikilink, чтобы владелец мог
проверить поиск по ним.

ВАЖНО: файлы создаются с распознаваемыми именами `__sba_проверка_*` —
их легко найти и удалить после проверки (скрипт печатает как).

Windows (PowerShell, из папки проекта):
    .\\.venv\\Scripts\\Activate.ps1
    python scripts\\smoke_sprint8.py

Скрипт сам берёт первую существующую папку из modules.rag.sources
(config/local.yaml). Можно указать папку вручную: --dir "C:\\Users\\...\\Notes".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sba.infra.config import load_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# уникальные метки — их и будем искать в Telegram (маловероятны в реальном корпусе)
TAG_TOKEN = "сбапроверка"
XLSX_TOKEN = "квартальныйбюджетсба2026"

NOTE_NAME = "__sba_проверка_obsidian.md"
XLSX_NAME = "__sba_проверка_бюджет.xlsx"

NOTE_TEXT = (
    "---\n"
    f"tags: [{TAG_TOKEN}, экспедицияладога]\n"
    "---\n"
    "# Тестовая заметка Sprint 8\n"
    "Проверяем поиск заметок Obsidian с учётом тегов.\n"
    "Связь на другую заметку: [[Снаряжение экспедиции]].\n"
    "Инлайновый тег #байдаркасба тоже должен попасть в поиск.\n"
)


def _pick_dir(explicit: str | None) -> Path:
    if explicit:
        target = Path(explicit).expanduser()
        if not target.is_dir():
            raise SystemExit(f"Папка не найдена: {target}")
        return target
    config = load_config(CONFIG_DIR)
    sources = config.modules.rag.sources
    if not sources:
        raise SystemExit(
            "Список modules.rag.sources пуст — задайте папки в config/local.yaml "
            "или укажите папку вручную: python scripts\\smoke_sprint8.py --dir \"...\""
        )
    for source in sources:
        path = Path(source).expanduser()
        if path.is_dir():
            return path
    raise SystemExit(f"Ни одна папка из modules.rag.sources не существует: {sources}")


def _write_xlsx(path: Path) -> None:
    import datetime as dt

    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Бюджет экспедиции"
    sheet.append(["Статья", "Сумма", "Срок"])
    sheet.append([XLSX_TOKEN, 100000, dt.datetime(2026, 8, 15)])
    sheet.append(["Резерв", 25000.0, None])
    workbook.save(str(path))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Тестовые файлы для проверки Sprint 8")
    parser.add_argument("--dir", help="папка для тестовых файлов (по умолчанию — из конфига)")
    target = _pick_dir(parser.parse_args().dir)

    note_path = target / NOTE_NAME
    xlsx_path = target / XLSX_NAME
    note_path.write_text(NOTE_TEXT, encoding="utf-8")
    _write_xlsx(xlsx_path)

    print("Тестовые файлы созданы:")
    print(f"  • {note_path}")
    print(f"  • {xlsx_path}")
    print()
    print("Дальше: запустите ассистента (python -m sba), подождите ~2 минуты")
    print("(индексатор подхватит новые файлы) и спросите в Telegram:")
    print(f'  1) «найди заметки с тегом {TAG_TOKEN}»  → должна найтись заметка')
    print(f'  2) «найди в документах {XLSX_TOKEN}»')
    print("     → должна найтись таблица (лист «Бюджет экспедиции»)")
    print()
    print("После проверки эти два файла можно удалить (имена начинаются с __sba_проверка).")


if __name__ == "__main__":
    main()

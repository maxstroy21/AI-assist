"""Файловые инструменты: чтение (список, поиск, чтение файлов) и операции
File Ops (Sprint 8: переместить/скопировать/переименовать/архив, дубликаты,
старые версии, undo).

Безопасность (FR-6.5, FR-9.5): работа строго внутри whitelisted-корней из
конфига (files.allowed_roots, RootGuard); пути наружу отклоняются после
resolve() (защита от .. и симлинков). Удаление — destructive: в корзину ОС
(send2trash) и только после подтверждения пользователя. Операции File Ops
по умолчанию идут в режиме сухого прогона (см. ops.py).
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from send2trash import send2trash

from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.infra.config import FilesConfig
from sba.modules.files.ops import FileOpsService
from sba.modules.files.safety import NO_ROOTS_HINT, RootGuard

SCAN_LIMIT = 50_000  # предохранитель от обхода гигантских деревьев
FIND_SHOWN = 30       # сколько путей кладём в ответ (потолок на объём → экономия токенов облака)
FIND_COUNT_CAP = 200  # докуда считаем всего совпадений; дальше — «более N» (не гоним обход зря)
SEARCH_TIME_BUDGET = 20.0  # секунд: OneDrive/сетевые папки перечисляются медленно


class PathArgs(BaseModel):
    path: str = Field(description="Путь к файлу или папке (внутри разрешённых папок)")


class ListArgs(BaseModel):
    path: str = Field(description="Путь к папке (внутри разрешённых папок)")
    pattern: str = Field(default="*", description="Маска имени, например *.pdf")


class FindArgs(BaseModel):
    name_pattern: str = Field(
        description="Имя файла или маска, например: отчёт.docx или *.log"
    )


class FilesToolset:
    def __init__(self, config: FilesConfig) -> None:
        self._config = config
        self._guard = RootGuard(config.allowed_roots)
        self._roots = self._guard.roots

    # ── безопасность путей (RootGuard, общий с файловыми операциями) ─────────

    def _roots_summary(self) -> str:
        return self._guard.summary()

    def _resolve(self, raw: str) -> Path:
        return self._guard.resolve(raw)

    # ── инструменты ──────────────────────────────────────────────────────────

    async def list_files(self, args: ListArgs) -> str:
        if args.path.strip().strip("'\"") in {"", ".", "/", "\\"}:
            if not self._roots:
                return NO_ROOTS_HINT
            return "Разрешённые папки:\n" + "\n".join(f"📁 {r}" for r in self._roots)
        target = self._resolve(args.path)
        if not target.exists():
            return f"Папка не существует: {target}. Разрешённые папки: {self._roots_summary()}"
        if not target.is_dir():
            return f"{target} — это файл, а не папка"
        entries = sorted(
            target.glob(args.pattern or "*"),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        if not entries:
            return f"В {target} нет ничего по маске {args.pattern!r}"
        limit = self._config.max_list_entries
        lines = [f"Содержимое {target}:"]
        for entry in entries[:limit]:
            try:
                stat = entry.stat()
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                if entry.is_dir():
                    lines.append(f"📁 {entry.name}/  (изменена {mtime})")
                else:
                    lines.append(f"📄 {entry.name}  ({stat.st_size:,} байт, изменён {mtime})")
            except OSError:
                lines.append(f"❓ {entry.name} (нет доступа)")
        if len(entries) > limit:
            lines.append(f"…и ещё {len(entries) - limit}")
        return "\n".join(lines)

    def _search(self, pattern: str, keep: int, count_cap: int) -> tuple[list[Path], int, bool]:
        """Рекурсивный поиск по имени во всех разрешённых корнях (в отдельном потоке).

        Возвращает (первые `keep` совпадений, всего_найдено, полный_ли_обход).
        Считаем совпадения до `count_cap`, но в список кладём лишь первые `keep`
        — тогда можно честно сказать «найдено X, показано Y», не раздувая ответ.
        Бюджет времени и лимит просмотренных файлов защищают от бесконечного
        перечисления OneDrive/сетевых папок; при их срабатывании полный_ли_обход
        = False (истинное «всего» неизвестно — покажем «максимум возможного»).
        """
        matches: list[Path] = []
        total = 0
        scanned = 0
        deadline = time.monotonic() + SEARCH_TIME_BUDGET
        pattern_lower = pattern.lower()
        for root in self._roots:
            for dirpath, dirnames, filenames in os.walk(root):
                if time.monotonic() > deadline:
                    return matches, total, False
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for filename in filenames:
                    scanned += 1
                    if scanned > SCAN_LIMIT:
                        return matches, total, False
                    if fnmatch.fnmatch(filename.lower(), pattern_lower):
                        total += 1
                        if len(matches) < keep:
                            matches.append(Path(dirpath) / filename)
                        if total >= count_cap:
                            # совпадений очень много — дальше считать незачем
                            return matches, total, True
        return matches, total, True

    @staticmethod
    def _normalize_pattern(pattern: str) -> str:
        # «.log» означает «файлы с расширением .log», а не файл с именем «.log»
        if pattern.startswith(".") and "*" not in pattern and "?" not in pattern:
            return "*" + pattern
        return pattern

    async def find_files(self, args: FindArgs) -> str:
        if not self._roots:
            return NO_ROOTS_HINT
        pattern = self._normalize_pattern(args.name_pattern.strip().strip("'\""))
        matches, total, complete = await asyncio.to_thread(
            self._search, pattern, FIND_SHOWN, FIND_COUNT_CAP
        )
        if not matches and complete and "*" not in pattern and "?" not in pattern:
            # точное имя не нашлось — ищем «содержит» (например, «debug» → *debug*)
            pattern = f"*{pattern}*"
            matches, total, complete = await asyncio.to_thread(
                self._search, pattern, FIND_SHOWN, FIND_COUNT_CAP
            )
        if not matches:
            note = "" if complete else " (обход прерван по лимиту времени — папки очень большие)"
            return (
                f"Ничего не найдено по маске {pattern!r} в разрешённых папках "
                f"({self._roots_summary()}){note}"
            )
        shown = len(matches)
        header = self._find_header(shown, total, complete)
        return header + "\n" + "\n".join(str(m) for m in matches)

    @staticmethod
    def _find_header(shown: int, total: int, complete: bool) -> str:
        """Честная строка о полноте выдачи (просьба владельца: «найдено X,
        показано Y», а если полный обход невозможен — «максимум возможного»)."""
        if not complete:
            # истинное «всего» неизвестно — обход прервался по лимиту времени/размера
            return (
                f"Показаны {shown} файлов — это максимум, который удалось собрать: "
                "папки очень большие и обход прерван по лимиту. "
                "Уточните маску, чтобы сузить поиск."
            )
        if total >= FIND_COUNT_CAP:
            return (
                f"Найдено более {FIND_COUNT_CAP} файлов, показаны первые {shown}. "
                "Уточните маску, чтобы сузить поиск."
            )
        if total > shown:
            return (
                f"Найдено {total}, показаны первые {shown} (ещё {total - shown}). "
                "Уточните маску, чтобы увидеть остальные."
            )
        return f"Найдено {total}:"

    async def read_document(self, args: PathArgs) -> str:
        raw = args.path.strip().strip("'\"")
        target = self._resolve(raw)
        if not target.exists():
            # голое имя без пути — попробуем найти файл сами
            if "/" not in raw and "\\" not in raw:
                matches, _total, _complete = await asyncio.to_thread(self._search, raw, 5, 5)
                if len(matches) == 1:
                    target = matches[0]
                elif matches:
                    return (
                        "Нашёл несколько файлов с таким именем — уточните путь:\n"
                        + "\n".join(str(m) for m in matches)
                    )
                else:
                    return (
                        f"Файл {raw!r} не найден в разрешённых папках "
                        f"({self._roots_summary()})"
                    )
            else:
                return f"Файл не существует: {target}"
        if target.is_dir():
            return f"{target} — папка; используйте list_files"
        raw_bytes = await asyncio.to_thread(target.read_bytes)
        if b"\x00" in raw_bytes[:1024]:
            return (
                f"{target.name} — не текстовый файл. Чтение PDF/DOCX/XLSX "
                "появится в следующих версиях (RAG, Sprint 4)."
            )
        text = raw_bytes.decode("utf-8", errors="replace")
        limit = self._config.max_read_chars
        header = f"Файл: {target}\n\n"
        if len(text) > limit:
            return (
                f"{header}{text[:limit]}\n"
                f"…(показаны первые {limit} символов из {len(text)})"
            )
        return header + text

    async def delete_file(self, args: PathArgs) -> str:
        target = self._resolve(args.path)
        if not target.exists():
            return f"Файл не существует: {target}"
        await asyncio.to_thread(send2trash, str(target))
        return f"Удалено в корзину: {target}"

    # ── регистрация ──────────────────────────────────────────────────────────

    def build_tools(self) -> list[ToolSpec]:
        roots_hint = (
            f" Разрешённые папки: {self._roots_summary()}." if self._roots else ""
        )
        return [
            ToolSpec(
                name="list_files",
                description="Показать содержимое папки: файлы и подпапки с размерами "
                f"и датами. path='.' покажет список разрешённых папок.{roots_hint}",
                args_schema=ListArgs,
                risk=RiskLevel.READ,
                module="files",
                handler=self.list_files,  # type: ignore[arg-type]
            ),
            ToolSpec(
                name="find_files",
                description="Найти файл по имени или маске во всех разрешённых папках, "
                "включая подпапки",
                args_schema=FindArgs,
                risk=RiskLevel.READ,
                module="files",
                handler=self.find_files,  # type: ignore[arg-type]
            ),
            ToolSpec(
                name="read_document",
                description="Прочитать текстовый файл (txt, md, csv, код). Если задано "
                "только имя без пути — файл ищется автоматически",
                args_schema=PathArgs,
                risk=RiskLevel.READ,
                module="files",
                handler=self.read_document,  # type: ignore[arg-type]
            ),
            ToolSpec(
                name="delete_file",
                description="Удалить файл или папку в корзину ОС (можно восстановить)",
                args_schema=PathArgs,
                risk=RiskLevel.DESTRUCTIVE,
                module="files",
                handler=self.delete_file,  # type: ignore[arg-type]
            ),
        ]


# ── File Ops (Sprint 8): операции с undo-журналом и анализ ───────────────────


class MoveArgs(BaseModel):
    src: str = Field(description="Что переместить: путь к файлу или папке")
    dst_dir: str = Field(
        description="Куда: папка назначения (создастся, если её ещё нет)"
    )


class CopyArgs(BaseModel):
    src: str = Field(description="Какой файл скопировать")
    dst_dir: str = Field(description="Куда: папка назначения")


class RenameArgs(BaseModel):
    path: str = Field(description="Файл или папка, которую переименовать")
    new_name: str = Field(
        description="Новое имя без пути, например: отчёт-2026.docx"
    )


class ArchiveArgs(BaseModel):
    folder: str = Field(description="Папка, файлы из которой убрать в архив")
    pattern: str = Field(default="*", description="Маска файлов, например *.pdf; * — все")
    older_than_days: int = Field(
        default=0, ge=0, description="Брать только файлы старше N дней; 0 — любые"
    )


class ScanArgs(BaseModel):
    path: str = Field(
        default="", description="Папка для проверки; пусто — все разрешённые папки"
    )


class UndoArgs(BaseModel):
    pass


def build_fileops_tools(service: FileOpsService) -> list[ToolSpec]:
    async def move_file(args: BaseModel) -> str:
        assert isinstance(args, MoveArgs)
        return await service.move(args.src, args.dst_dir)

    async def copy_file(args: BaseModel) -> str:
        assert isinstance(args, CopyArgs)
        return await service.copy(args.src, args.dst_dir)

    async def rename_file(args: BaseModel) -> str:
        assert isinstance(args, RenameArgs)
        return await service.rename(args.path, args.new_name)

    async def archive_files(args: BaseModel) -> str:
        assert isinstance(args, ArchiveArgs)
        return await service.archive(args.folder, args.pattern, args.older_than_days)

    async def find_duplicates(args: BaseModel) -> str:
        assert isinstance(args, ScanArgs)
        return await service.duplicates_text(args.path)

    async def find_old_versions(args: BaseModel) -> str:
        assert isinstance(args, ScanArgs)
        return await service.old_versions_text(args.path)

    async def undo_file_operation(args: BaseModel) -> str:
        assert isinstance(args, UndoArgs)
        return await service.undo_last()

    # в сухом прогоне операции ничего не меняют (только план) — риск write,
    # чтобы не спрашивать подтверждение на безвредный план; в боевом режиме
    # массовое архивирование — destructive (docs/04 §11), одиночные операции
    # обратимы через undo и не перезаписывают — write (ADR-10)
    dry = not service.executes
    mode_note = (
        " Сейчас режим сухого прогона: составляется только план, файлы не меняются."
        if dry
        else " Без перезаписи; отменить можно через undo_file_operation."
    )
    return [
        ToolSpec(
            name="move_file",
            description="Переместить файл или папку в другую папку." + mode_note,
            args_schema=MoveArgs,
            risk=RiskLevel.WRITE,
            module="files",
            handler=move_file,
        ),
        ToolSpec(
            name="copy_file",
            description="Скопировать файл в другую папку." + mode_note,
            args_schema=CopyArgs,
            risk=RiskLevel.WRITE,
            module="files",
            handler=copy_file,
        ),
        ToolSpec(
            name="rename_file",
            description="Переименовать файл или папку (имя меняется, папка та же)."
            + mode_note,
            args_schema=RenameArgs,
            risk=RiskLevel.WRITE,
            module="files",
            handler=rename_file,
        ),
        ToolSpec(
            name="archive_files",
            description="Убрать файлы папки в её подпапку-архив (по маске и/или "
            "старше N дней)." + mode_note,
            args_schema=ArchiveArgs,
            risk=RiskLevel.WRITE if dry else RiskLevel.DESTRUCTIVE,
            module="files",
            handler=archive_files,
        ),
        ToolSpec(
            name="find_duplicates",
            description="Найти побайтово одинаковые файлы (дубликаты) и показать, "
            "сколько места они занимают",
            args_schema=ScanArgs,
            risk=RiskLevel.READ,
            module="files",
            handler=find_duplicates,
        ),
        ToolSpec(
            name="find_old_versions",
            description="Найти старые версии документов по похожим именам "
            "(_v2, (1), копия, даты) и датам изменения",
            args_schema=ScanArgs,
            risk=RiskLevel.READ,
            module="files",
            handler=find_old_versions,
        ),
        ToolSpec(
            name="undo_file_operation",
            description="Откатить последнюю выполненную файловую операцию "
            "(перемещение/копирование/переименование/архив)",
            args_schema=UndoArgs,
            risk=RiskLevel.WRITE,
            module="files",
            handler=undo_file_operation,
        ),
    ]

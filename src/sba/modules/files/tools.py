"""Файловые инструменты v1: список, чтение текстовых файлов, удаление в корзину.

Безопасность (FR-6.5, FR-9.5): работа строго внутри whitelisted-корней из
конфига (files.allowed_roots); пути наружу отклоняются после resolve()
(защита от .. и симлинков). Удаление — destructive: в корзину ОС
(send2trash) и только после подтверждения пользователя.
Полноценный File Ops (перемещение, дубликаты, undo) — Sprint 8.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from send2trash import send2trash

from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.infra.config import FilesConfig

SCAN_LIMIT = 50_000  # предохранитель от многоминутного обхода гигантских деревьев
FIND_MAX_MATCHES = 30

NO_ROOTS_HINT = (
    "Файловые инструменты не настроены: список разрешённых папок пуст. "
    "Попросите владельца добавить в config/local.yaml:\n"
    "files:\n  allowed_roots: ['C:\\Users\\имя\\Documents']"
)


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
        self._roots = [Path(r).expanduser().resolve() for r in config.allowed_roots]

    # ── безопасность путей ───────────────────────────────────────────────────

    def _roots_summary(self) -> str:
        return ", ".join(str(r) for r in self._roots)

    def _resolve(self, raw: str) -> Path:
        if not self._roots:
            raise ValueError(NO_ROOTS_HINT)
        cleaned = raw.strip().strip("'\"")
        # «Downloads» должно означать сам разрешённый корень с таким именем,
        # а не подпапку Downloads внутри корней
        for root in self._roots:
            if cleaned.rstrip("\\/").lower() in (root.name.lower(), str(root).lower()):
                return root
        path = Path(cleaned).expanduser()
        candidates = [path] if path.is_absolute() else [root / path for root in self._roots]
        for candidate in candidates:
            resolved = candidate.resolve()
            if any(resolved.is_relative_to(root) for root in self._roots):
                if resolved.exists() or candidate is candidates[-1]:
                    return resolved
        raise ValueError(
            f"путь {raw!r} вне разрешённых папок. Разрешены: {self._roots_summary()}"
        )

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

    def _search(self, pattern: str, limit: int) -> list[Path]:
        """Рекурсивный поиск по имени во всех разрешённых корнях (в отдельном потоке)."""
        matches: list[Path] = []
        scanned = 0
        pattern_lower = pattern.lower()
        for root in self._roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for filename in filenames:
                    scanned += 1
                    if scanned > SCAN_LIMIT:
                        return matches
                    if fnmatch.fnmatch(filename.lower(), pattern_lower):
                        matches.append(Path(dirpath) / filename)
                        if len(matches) >= limit:
                            return matches
        return matches

    async def find_files(self, args: FindArgs) -> str:
        if not self._roots:
            return NO_ROOTS_HINT
        pattern = args.name_pattern.strip().strip("'\"")
        matches = await asyncio.to_thread(self._search, pattern, FIND_MAX_MATCHES)
        if not matches:
            return (
                f"Ничего не найдено по маске {pattern!r} в разрешённых папках "
                f"({self._roots_summary()})"
            )
        lines = [f"Найдено {len(matches)}:"] + [str(m) for m in matches]
        if len(matches) >= FIND_MAX_MATCHES:
            lines.append("…(показаны первые совпадения, уточните маску)")
        return "\n".join(lines)

    async def read_document(self, args: PathArgs) -> str:
        raw = args.path.strip().strip("'\"")
        target = self._resolve(raw)
        if not target.exists():
            # голое имя без пути — попробуем найти файл сами
            if "/" not in raw and "\\" not in raw:
                matches = await asyncio.to_thread(self._search, raw, 5)
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

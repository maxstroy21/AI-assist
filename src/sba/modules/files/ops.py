"""File Ops (Sprint 8): переместить/скопировать/переименовать/архивировать
с undo-журналом.

СУХОЙ ПРОГОН по умолчанию — защита самого опасного функционала (риск Sprint 8,
порча файлов): операции только планируются и журналируются, файловая система
не меняется. Реальное исполнение включается в local.yaml
(modules.fileops.execute: true) после недели проверки планов владельцем.

Правила безопасности:
- строго внутри разрешённых корней (RootGuard — общий с файловыми тулзами);
- перезапись запрещена всегда: занятая цель — отказ, а не замена;
- выполненные операции обратимы: undo откатывает последний выполненный батч
  (перемещения — обратно, созданная копия — в корзину ОС).

Тексты результатов адресованы модели (урок Sprint 2/5 — 7B фабрикует отчёты
о действиях): в сухом прогоне они прямо требуют сказать владельцу, что
НИЧЕГО не выполнено.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from datetime import datetime
from pathlib import Path

import structlog
from send2trash import send2trash

from sba.infra.config import FileOpsConfig
from sba.modules.files.analysis import find_duplicate_groups, find_version_groups
from sba.modules.files.opsstore import EntryView, FileOpsStore
from sba.modules.files.safety import NO_ROOTS_HINT, RootGuard

log = structlog.get_logger(__name__)

PLAN_PREVIEW_LINES = 15  # сколько строк плана показывать при массовой операции

INCOMPLETE_NOTE = (
    "\n…(обход прерван по лимиту времени или числа файлов — результат может "
    "быть неполным; сузьте папку)"
)


def _fmt_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} ГБ"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} МБ"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} КБ"
    return f"{n} байт"


def _fmt_day(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


class FileOpsService:
    def __init__(self, store: FileOpsStore, guard: RootGuard, config: FileOpsConfig) -> None:
        self._store = store
        self._guard = guard
        self._config = config

    @property
    def executes(self) -> bool:
        return self._config.execute

    @property
    def guard(self) -> RootGuard:
        return self._guard

    # ── общие проверки и низкоуровневое исполнение ───────────────────────────

    @staticmethod
    def _check_pair(src: Path, dst: Path) -> str | None:
        """Общие правила операции; текст отказа или None (можно выполнять)."""
        if not src.exists():
            return f"Отказ: {src} не существует"
        if dst.exists():
            return f"Отказ: цель уже существует, перезапись запрещена: {dst}"
        if src == dst:
            return "Отказ: источник и цель совпадают"
        if src.is_dir() and dst.is_relative_to(src):
            return "Отказ: нельзя переместить папку внутрь самой себя"
        return None

    @staticmethod
    def _perform(op: str, src: Path, dst: Path) -> str | None:
        """Одна операция на диске; текст ошибки или None. Только для execute."""
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if op == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
        except OSError as exc:
            return str(exc)
        return None

    def _dry_plan(self, lines: list[str]) -> str:
        return (
            "🧪 СУХОЙ ПРОГОН: операция НЕ выполнена, составлен только план.\n"
            + "\n".join(lines)
            + "\nФайлы не тронуты. Сообщи владельцу, что это только план — "
            "исполнение отключено на время проверки (включается настройкой "
            "modules.fileops.execute в local.yaml)."
        )

    async def _single(self, kind: str, verb: str, op: str, src: Path, dst: Path) -> str:
        description = f"{verb} {src} → {dst}"
        if not self.executes:
            await self._store.create_batch(
                kind, dry_run=True, description=description,
                entries=[(op, str(src), str(dst))],
            )
            log.info("fileops_dry_run", kind=kind, src=str(src), dst=str(dst))
            return self._dry_plan([f"План: {description}"])
        batch_id = await self._store.create_batch(
            kind, dry_run=False, description=description,
            entries=[(op, str(src), str(dst))],
        )
        error = await asyncio.to_thread(self._perform, op, src, dst)
        await self._store.mark_entry(batch_id, 0, error=error)
        if error is not None:
            log.error("fileops_failed", kind=kind, src=str(src), error=error)
            return f"❌ Не получилось {verb}: {error}"
        log.info("fileops_done", kind=kind, src=str(src), dst=str(dst))
        return (
            f"✅ Выполнено: {description}.\n"
            "Отменить можно инструментом undo_file_operation."
        )

    # ── операции ─────────────────────────────────────────────────────────────

    async def move(self, src_raw: str, dst_dir_raw: str) -> str:
        try:
            src = self._guard.resolve(src_raw)
            dst_dir = self._guard.resolve(dst_dir_raw)
        except ValueError as exc:
            return f"Отказ: {exc}"
        if dst_dir.exists() and not dst_dir.is_dir():
            return f"Отказ: {dst_dir} — файл, а не папка назначения"
        dst = dst_dir / src.name
        refusal = self._check_pair(src, dst)
        if refusal is not None:
            return refusal
        return await self._single("move", "переместить", "move", src, dst)

    async def copy(self, src_raw: str, dst_dir_raw: str) -> str:
        try:
            src = self._guard.resolve(src_raw)
            dst_dir = self._guard.resolve(dst_dir_raw)
        except ValueError as exc:
            return f"Отказ: {exc}"
        if src.is_dir():
            return "Отказ: копирование папок не поддерживается — укажите файл"
        if dst_dir.exists() and not dst_dir.is_dir():
            return f"Отказ: {dst_dir} — файл, а не папка назначения"
        dst = dst_dir / src.name
        refusal = self._check_pair(src, dst)
        if refusal is not None:
            return refusal
        return await self._single("copy", "скопировать", "copy", src, dst)

    async def rename(self, path_raw: str, new_name: str) -> str:
        name = new_name.strip().strip("'\"")
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            return f"Отказ: недопустимое новое имя {new_name!r} — нужно имя без пути"
        try:
            src = self._guard.resolve(path_raw)
        except ValueError as exc:
            return f"Отказ: {exc}"
        dst = src.parent / name
        refusal = self._check_pair(src, dst)
        if refusal is not None:
            return refusal
        return await self._single("rename", "переименовать", "move", src, dst)

    async def archive(
        self, folder_raw: str, pattern: str = "*", older_than_days: int = 0
    ) -> str:
        try:
            folder = self._guard.resolve(folder_raw)
        except ValueError as exc:
            return f"Отказ: {exc}"
        if not folder.is_dir():
            return f"Отказ: {folder} — не папка"
        # «.pdf» означает «файлы с расширением .pdf» (как в find_files)
        mask = pattern.strip().strip("'\"") or "*"
        if mask.startswith(".") and "*" not in mask and "?" not in mask:
            mask = "*" + mask
        archive_dir = folder / self._config.archive_subdir
        cutoff = time.time() - older_than_days * 86400 if older_than_days > 0 else None
        candidates: list[Path] = []
        for p in sorted(folder.glob(mask), key=lambda p: p.name.lower()):
            if not p.is_file():
                continue
            try:
                if cutoff is not None and p.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            candidates.append(p)
        if not candidates:
            age = f" старше {older_than_days} дн." if cutoff is not None else ""
            return f"Архивировать нечего: в {folder} нет файлов по маске {mask!r}{age}"
        if len(candidates) > self._config.max_batch_files:
            return (
                f"Отказ: под маску {mask!r} попадает {len(candidates)} файлов — "
                f"больше предела {self._config.max_batch_files}. Сузьте маску или "
                "добавьте фильтр older_than_days."
            )
        # план назначений с уникализацией имён внутри архива
        planned: set[str] = set()
        entries: list[tuple[str, str, str]] = []
        for src in candidates:
            dst = archive_dir / src.name
            n = 2
            while str(dst) in planned or dst.exists():
                dst = archive_dir / f"{src.stem} ({n}){src.suffix}"
                n += 1
            planned.add(str(dst))
            entries.append(("move", str(src), str(dst)))
        description = f"архив: {len(entries)} файл(ов) из {folder} → {archive_dir}"
        preview = [f"• {src} → {dst}" for _, src, dst in entries[:PLAN_PREVIEW_LINES]]
        if len(entries) > PLAN_PREVIEW_LINES:
            preview.append(f"…и ещё {len(entries) - PLAN_PREVIEW_LINES}")
        if not self.executes:
            await self._store.create_batch(
                "archive", dry_run=True, description=description, entries=entries
            )
            log.info("fileops_dry_run", kind="archive", files=len(entries))
            return self._dry_plan([f"План ({description}):", *preview])
        batch_id = await self._store.create_batch(
            "archive", dry_run=False, description=description, entries=entries
        )
        failures: list[str] = []
        for seq, (op, src_s, dst_s) in enumerate(entries):
            error = await asyncio.to_thread(self._perform, op, Path(src_s), Path(dst_s))
            await self._store.mark_entry(batch_id, seq, error=error)
            if error is not None:
                failures.append(f"⚠️ {src_s}: {error}")
        moved = len(entries) - len(failures)
        log.info("fileops_done", kind="archive", moved=moved, failed=len(failures))
        text = f"✅ В архив {archive_dir} перемещено {moved} из {len(entries)} файлов."
        if failures:
            text += "\n" + "\n".join(failures[:10])
        text += "\nОтменить можно инструментом undo_file_operation."
        return text

    # ── откат ────────────────────────────────────────────────────────────────

    @staticmethod
    def _undo_one(entry: EntryView) -> str | None:
        """Обратная операция для одной выполненной записи; ошибка или None."""
        try:
            if entry.op == "copy":
                # откат копирования — удалить созданную копию (в корзину ОС)
                if not Path(entry.dst).exists():
                    return "копия уже отсутствует"
                send2trash(entry.dst)
                return None
            dst, src = Path(entry.dst), Path(entry.src)
            if not dst.exists():
                return "файла уже нет на новом месте"
            if src.exists():
                return "исходное место уже занято"
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
        except OSError as exc:
            return str(exc)
        return None

    async def undo_last(self) -> str:
        batch = await self._store.last_undoable()
        if batch is None:
            note = (
                " Сейчас сухой прогон: операции только планируются, "
                "откатывать нечего." if not self.executes else ""
            )
            return "Отменять нечего: выполненных файловых операций не найдено." + note
        entries = await self._store.entries(batch.id)
        todo = [e for e in entries if e.executed][::-1]  # откат в обратном порядке
        if not todo:
            await self._store.mark_undone(batch.id)
            return (
                "В последней операции не было успешно выполненных действий — "
                "откатывать нечего."
            )
        undo_id = await self._store.create_batch(
            "undo",
            dry_run=False,
            description=f"отмена: {batch.description}",
            entries=[
                ("trash" if e.op == "copy" else "move", e.dst, e.src) for e in todo
            ],
            undo_of=batch.id,
        )
        failures: list[str] = []
        for seq, entry in enumerate(todo):
            error = await asyncio.to_thread(self._undo_one, entry)
            await self._store.mark_entry(undo_id, seq, error=error)
            if error is not None:
                failures.append(f"⚠️ {entry.dst}: {error}")
        await self._store.mark_undone(batch.id)
        ok = len(todo) - len(failures)
        log.info("fileops_undone", batch=batch.id, ok=ok, failed=len(failures))
        text = f"↩️ Отмена «{batch.description}»: возвращено {ok} из {len(todo)}."
        if failures:
            text += "\n" + "\n".join(failures[:10])
        return text

    # ── анализ: дубликаты и старые версии (только чтение) ────────────────────

    def _scan_roots(self, path_raw: str) -> list[Path] | str:
        """Корни сканирования: заданная папка или все разрешённые; текст — отказ."""
        if path_raw.strip():
            try:
                target = self._guard.resolve(path_raw)
            except ValueError as exc:
                return f"Отказ: {exc}"
            if not target.is_dir():
                return f"Отказ: {target} — не папка"
            return [target]
        if not self._guard.roots:
            return NO_ROOTS_HINT
        return self._guard.roots

    async def duplicates_text(self, path_raw: str = "") -> str:
        roots = self._scan_roots(path_raw)
        if isinstance(roots, str):
            return roots
        groups, complete = await asyncio.to_thread(
            find_duplicate_groups,
            roots,
            limit_files=self._config.scan_limit_files,
            time_budget=self._config.time_budget_seconds,
        )
        where = ", ".join(str(r) for r in roots)
        if not groups:
            return f"Побайтовых дубликатов в {where} не найдено." + (
                "" if complete else INCOMPLETE_NOTE
            )
        shown = groups[: self._config.max_groups]
        wasted = sum(g.wasted_bytes for g in groups)
        lines = [
            f"Найдено групп дубликатов: {len(groups)} "
            f"(лишних данных ~{_fmt_size(wasted)}). Перечисляй только эти файлы:"
        ]
        for group in shown:
            lines.append(
                f"— {len(group.paths)} шт. × {_fmt_size(group.size)} "
                f"(лишних {_fmt_size(group.wasted_bytes)}):"
            )
            lines += [f"   {p}" for p in group.paths]
        if len(groups) > len(shown):
            lines.append(f"…и ещё {len(groups) - len(shown)} групп (сузьте папку)")
        if not complete:
            lines.append(INCOMPLETE_NOTE.strip())
        return "\n".join(lines)

    async def old_versions_text(self, path_raw: str = "") -> str:
        roots = self._scan_roots(path_raw)
        if isinstance(roots, str):
            return roots
        groups, complete = await asyncio.to_thread(
            find_version_groups,
            roots,
            limit_files=self._config.scan_limit_files,
            time_budget=self._config.time_budget_seconds,
        )
        where = ", ".join(str(r) for r in roots)
        if not groups:
            return f"Похожих на версии одного документа файлов в {where} не найдено." + (
                "" if complete else INCOMPLETE_NOTE
            )
        shown = groups[: self._config.max_groups]
        lines = [
            f"Найдено групп похожих версий: {len(groups)}. Это эвристика по именам "
            "и датам — решает владелец. Перечисляй только эти файлы:"
        ]
        for group in shown:
            lines.append(f"— «{group.key}»: актуальная (по дате) {group.newest}")
            lines += [
                f"   старее ({_fmt_day(mtime)}): {p}" for p, mtime in group.older
            ]
        if len(groups) > len(shown):
            lines.append(f"…и ещё {len(groups) - len(shown)} групп (сузьте папку)")
        if not complete:
            lines.append(INCOMPLETE_NOTE.strip())
        return "\n".join(lines)

    # ── обзор для /fileops (мимо LLM) ────────────────────────────────────────

    async def overview_text(self) -> str:
        mode = (
            "⚡ боевой режим: операции ВЫПОЛНЯЮТСЯ"
            if self.executes
            else "🧪 сухой прогон: операции только планируются, файлы не меняются"
        )
        lines = [f"Файловые операции — {mode}."]
        batches = await self._store.recent()
        if not batches:
            lines.append("Журнал пуст: операций ещё не было.")
            return "\n".join(lines)
        lines.append("Последние операции (новые сверху):")
        for b in batches:
            entries = await self._store.entries(b.id)
            done = sum(e.executed for e in entries)
            if b.dry_run:
                status = "план"
            elif b.undone_at:
                status = "отменена"
            else:
                status = f"{done}/{len(entries)} выполнено"
            lines.append(
                f"• {b.created_at[:16]} UTC | {b.kind} | {status} | {b.description[:80]}"
            )
        return "\n".join(lines)

"""Анализ файлов (Sprint 8): побайтовые дубликаты и старые версии документов.

Чистые синхронные функции — вызываются через asyncio.to_thread. Обход и
хэширование ограничены бюджетом времени и потолком числа файлов: OneDrive
и сетевые папки перечисляются медленно, инструмент не должен вешать agent
loop (урок «всё внешнее обязано иметь таймаут»).

Дубликаты — только чтение: сравнение по размеру, затем sha256 (побайтово,
FR-6.3). «Почти дубликаты» по векторной близости — следующий шаг Sprint 8.
Старые версии (FR-6.4) — эвристики имени (_v2, (1), копия, даты) + mtime.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

HASH_CHUNK = 1 << 20  # 1 МБ: хэшируем кусками, между кусками проверяем бюджет

# Хвосты имён, означающие версию/копию; срезаются повторно до устойчивого имени:
# «отчёт - копия (2)» → «отчёт - копия» → «отчёт»
_VERSION_SUFFIXES = (
    re.compile(r"\s*\(\d+\)$"),                                   # « (1)», «(2)»
    re.compile(r"[ _\-.]*(?:v|ver|версия|version)\.?\s*\d+$", re.IGNORECASE),
    re.compile(
        r"[ _\-.]*(?:копия|copy|final|финал|итог(?:овый|овая)?|new|новый|новая"
        r"|old|старый|старая|бэкап|backup)$",
        re.IGNORECASE,
    ),
    re.compile(r"[ _\-.]*\d{4}[-._]\d{2}[-._]\d{2}$"),            # 2026-07-21
    re.compile(r"[ _\-.]*\d{2}[-._]\d{2}[-._]\d{4}$"),            # 21.07.2026
    re.compile(r"[ _\-.]*\d{6,8}$"),                              # 20260721
    re.compile(r"[ _\-.]+$"),
)


@dataclass(frozen=True)
class DuplicateGroup:
    size: int                  # размер одного экземпляра, байт
    paths: tuple[Path, ...]    # одинаковые побайтово файлы

    @property
    def wasted_bytes(self) -> int:
        return self.size * (len(self.paths) - 1)


@dataclass(frozen=True)
class VersionGroup:
    key: str                              # нормализованное имя + расширение
    newest: Path                          # самая свежая версия (по mtime)
    older: tuple[tuple[Path, float], ...]  # старые версии: (путь, mtime), новые сверху


def normalize_stem(stem: str) -> str:
    """Имя файла без версионных хвостов: «Отчёт_v2 (1)» → «отчёт»."""
    result = stem.strip().lower()
    changed = True
    while changed:
        changed = False
        for rx in _VERSION_SUFFIXES:
            trimmed = rx.sub("", result)
            if trimmed != result and trimmed:
                result = trimmed
                changed = True
    return result


def _walk_files(
    roots: list[Path], limit: int, deadline: float
) -> tuple[list[Path], bool]:
    """Все файлы корней (скрытые папки пропускаются); (файлы, полный_ли_обход)."""
    files: list[Path] = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if time.monotonic() > deadline:
                return files, False
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                files.append(Path(dirpath) / filename)
                if len(files) >= limit:
                    return files, False
    return files, True


def _sha256(path: Path, deadline: float) -> str | None:
    """Хэш файла; None — не успели по бюджету или файл недоступен."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while chunk := fh.read(HASH_CHUNK):
                if time.monotonic() > deadline:
                    return None
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def find_duplicate_groups(
    roots: list[Path], *, limit_files: int, time_budget: float
) -> tuple[list[DuplicateGroup], bool]:
    """Группы побайтово одинаковых файлов, отсортированы по лишним байтам."""
    deadline = time.monotonic() + time_budget
    files, complete = _walk_files(roots, limit_files, deadline)
    by_size: dict[int, list[Path]] = {}
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0:  # пустые файлы — не дубликаты по смыслу
            continue
        by_size.setdefault(size, []).append(path)

    groups: list[DuplicateGroup] = []
    for size, same_size in by_size.items():
        if len(same_size) < 2:
            continue
        if time.monotonic() > deadline:
            complete = False
            break
        by_hash: dict[str, list[Path]] = {}
        for path in same_size:
            digest = _sha256(path, deadline)
            if digest is None:
                complete = complete and time.monotonic() <= deadline
                continue
            by_hash.setdefault(digest, []).append(path)
        for paths in by_hash.values():
            if len(paths) > 1:
                groups.append(DuplicateGroup(size=size, paths=tuple(sorted(paths))))
    groups.sort(key=lambda g: g.wasted_bytes, reverse=True)
    return groups, complete


def find_version_groups(
    roots: list[Path], *, limit_files: int, time_budget: float
) -> tuple[list[VersionGroup], bool]:
    """Группы «похоже, версии одного документа»: одинаковое нормализованное имя
    и расширение; свежая по mtime — актуальная, остальные — кандидаты в архив."""
    deadline = time.monotonic() + time_budget
    files, complete = _walk_files(roots, limit_files, deadline)
    by_key: dict[str, list[tuple[Path, float]]] = {}
    for path in files:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        key = normalize_stem(path.stem) + path.suffix.lower()
        by_key.setdefault(key, []).append((path, mtime))

    groups: list[VersionGroup] = []
    for key, members in by_key.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: m[1], reverse=True)
        newest = members[0][0]
        groups.append(VersionGroup(key=key, newest=newest, older=tuple(members[1:])))
    # больше всего версий — интереснее всего для разбора
    groups.sort(key=lambda g: len(g.older), reverse=True)
    return groups, complete

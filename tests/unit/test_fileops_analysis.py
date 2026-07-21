"""Анализ файлов: дубликаты (размер → sha256) и старые версии (имена + mtime)."""

import os
import time
from pathlib import Path

from sba.modules.files.analysis import (
    find_duplicate_groups,
    find_version_groups,
    normalize_stem,
)


def test_duplicates_found_by_content(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("одинаковое содержимое", encoding="utf-8")
    sub = tmp_path / "папка"
    sub.mkdir()
    (sub / "b.txt").write_text("одинаковое содержимое", encoding="utf-8")
    (tmp_path / "другой.txt").write_text("другое", encoding="utf-8")

    groups, complete = find_duplicate_groups(
        [tmp_path], limit_files=1000, time_budget=10.0
    )
    assert complete
    assert len(groups) == 1
    assert {p.name for p in groups[0].paths} == {"a.txt", "b.txt"}
    assert groups[0].wasted_bytes == groups[0].size


def test_same_size_different_content_not_duplicates(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"aaaa")
    (tmp_path / "b.txt").write_bytes(b"bbbb")
    groups, _ = find_duplicate_groups([tmp_path], limit_files=1000, time_budget=10.0)
    assert groups == []


def test_empty_files_ignored(tmp_path: Path) -> None:
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.txt").touch()
    groups, _ = find_duplicate_groups([tmp_path], limit_files=1000, time_budget=10.0)
    assert groups == []


def test_file_limit_marks_incomplete(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(str(i), encoding="utf-8")
    _groups, complete = find_duplicate_groups(
        [tmp_path], limit_files=3, time_budget=10.0
    )
    assert not complete


def test_normalize_stem_strips_version_markers() -> None:
    assert normalize_stem("Отчёт_v2") == "отчёт"
    assert normalize_stem("отчёт (1)") == "отчёт"
    assert normalize_stem("отчёт - копия (2)") == "отчёт"
    assert normalize_stem("доклад_2026-07-21") == "доклад"
    assert normalize_stem("доклад 20260721") == "доклад"
    assert normalize_stem("план final") == "план"
    # хвост не срезается, если после него ничего не остаётся
    assert normalize_stem("v2") == "v2"
    # обычные имена не меняются (4 цифры года — не дата)
    assert normalize_stem("бюджет 2026") == "бюджет 2026"


def test_version_groups_newest_by_mtime(tmp_path: Path) -> None:
    old = tmp_path / "отчёт_v1.docx"
    old.write_text("v1", encoding="utf-8")
    week_ago = time.time() - 7 * 86400
    os.utime(old, (week_ago, week_ago))
    new = tmp_path / "отчёт_v2.docx"
    new.write_text("v2", encoding="utf-8")
    (tmp_path / "другое.docx").write_text("x", encoding="utf-8")

    groups, complete = find_version_groups(
        [tmp_path], limit_files=1000, time_budget=10.0
    )
    assert complete
    assert len(groups) == 1
    assert groups[0].newest == new
    assert [p for p, _ in groups[0].older] == [old]


def test_version_groups_different_extensions_not_grouped(tmp_path: Path) -> None:
    (tmp_path / "отчёт.docx").write_text("x", encoding="utf-8")
    (tmp_path / "отчёт.pdf").write_text("x", encoding="utf-8")
    groups, _ = find_version_groups([tmp_path], limit_files=1000, time_budget=10.0)
    assert groups == []

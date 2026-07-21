"""File Ops (Sprint 8): сухой прогон, исполнение, undo, запрет перезаписи."""

from pathlib import Path

import pytest

import sba.modules.files.ops as ops_module
from sba.infra.config import FileOpsConfig
from sba.infra.db import Database
from sba.modules.files.ops import FileOpsService
from sba.modules.files.opsstore import FileOpsStore
from sba.modules.files.safety import RootGuard


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_service(db: Database, root: Path, execute: bool = False, **overrides) -> FileOpsService:
    config = FileOpsConfig(execute=execute, **overrides)
    return FileOpsService(FileOpsStore(db), RootGuard([root]), config)


# ── сухой прогон (режим по умолчанию) ────────────────────────────────────────


async def test_dry_run_move_does_not_touch_files(db: Database, tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path)

    result = await service.move(str(src), str(tmp_path / "куда"))
    assert "СУХОЙ ПРОГОН" in result
    assert "НЕ выполнена" in result
    assert src.exists()                       # файл на месте
    assert not (tmp_path / "куда").exists()   # папка не создана


async def test_dry_run_is_journaled_as_plan(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path)
    await service.move(str(tmp_path / "a.txt"), str(tmp_path / "куда"))

    store = FileOpsStore(db)
    batches = await store.recent()
    assert len(batches) == 1
    assert batches[0].dry_run is True
    assert await store.last_undoable() is None   # план откату не подлежит


async def test_dry_run_undo_says_nothing_to_revert(db: Database, tmp_path: Path) -> None:
    service = make_service(db, tmp_path)
    result = await service.undo_last()
    assert "Отменять нечего" in result
    assert "сухой прогон" in result.lower()


async def test_dry_run_archive_plans_all_files(db: Database, tmp_path: Path) -> None:
    for name in ("a.pdf", "b.pdf", "c.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path)

    result = await service.archive(str(tmp_path), pattern="*.pdf")
    assert "СУХОЙ ПРОГОН" in result
    assert "a.pdf" in result and "b.pdf" in result
    assert "c.txt" not in result
    assert (tmp_path / "a.pdf").exists()
    assert not (tmp_path / "_архив").exists()


# ── исполнение (execute=true) ────────────────────────────────────────────────


async def test_execute_move_and_undo(db: Database, tmp_path: Path) -> None:
    src = tmp_path / "a.txt"
    src.write_text("данные", encoding="utf-8")
    dst_dir = tmp_path / "куда"
    service = make_service(db, tmp_path, execute=True)

    result = await service.move(str(src), str(dst_dir))
    assert "✅" in result
    assert not src.exists()
    assert (dst_dir / "a.txt").read_text(encoding="utf-8") == "данные"

    undone = await service.undo_last()
    assert "возвращено 1 из 1" in undone
    assert src.exists()
    assert not (dst_dir / "a.txt").exists()


async def test_execute_rename_and_undo(db: Database, tmp_path: Path) -> None:
    src = tmp_path / "старое.txt"
    src.write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path, execute=True)

    result = await service.rename(str(src), "новое.txt")
    assert "✅" in result
    assert (tmp_path / "новое.txt").exists()

    await service.undo_last()
    assert src.exists()
    assert not (tmp_path / "новое.txt").exists()


async def test_execute_copy_undo_trashes_copy(
    db: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "a.txt"
    src.write_text("x", encoding="utf-8")
    dst_dir = tmp_path / "копии"
    service = make_service(db, tmp_path, execute=True)

    await service.copy(str(src), str(dst_dir))
    copy_path = dst_dir / "a.txt"
    assert copy_path.exists()
    assert src.exists()

    trashed: list[str] = []

    def fake_trash(path: str) -> None:
        trashed.append(path)
        Path(path).unlink()

    monkeypatch.setattr(ops_module, "send2trash", fake_trash)
    undone = await service.undo_last()
    assert "возвращено 1 из 1" in undone
    assert trashed == [str(copy_path)]
    assert src.exists()  # оригинал не тронут


async def test_execute_archive_and_undo(db: Database, tmp_path: Path) -> None:
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path, execute=True)

    result = await service.archive(str(tmp_path), pattern="*.pdf")
    assert "перемещено 2 из 2" in result
    archive = tmp_path / "_архив"
    assert (archive / "a.pdf").exists() and (archive / "b.pdf").exists()

    undone = await service.undo_last()
    assert "возвращено 2 из 2" in undone
    assert (tmp_path / "a.pdf").exists() and (tmp_path / "b.pdf").exists()


async def test_undo_twice_has_nothing_second_time(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path, execute=True)
    await service.move(str(tmp_path / "a.txt"), str(tmp_path / "куда"))
    await service.undo_last()
    result = await service.undo_last()
    assert "Отменять нечего" in result


async def test_archive_name_collision_gets_suffix(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("новый", encoding="utf-8")
    archive = tmp_path / "_архив"
    archive.mkdir()
    (archive / "a.txt").write_text("старый", encoding="utf-8")
    service = make_service(db, tmp_path, execute=True)

    await service.archive(str(tmp_path), pattern="a.txt")
    assert (archive / "a.txt").read_text(encoding="utf-8") == "старый"  # не перезаписан
    assert (archive / "a (2).txt").read_text(encoding="utf-8") == "новый"


# ── отказы (одинаковы в обоих режимах) ───────────────────────────────────────


async def test_overwrite_refused(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("новый", encoding="utf-8")
    dst_dir = tmp_path / "куда"
    dst_dir.mkdir()
    (dst_dir / "a.txt").write_text("старый", encoding="utf-8")
    service = make_service(db, tmp_path, execute=True)

    result = await service.move(str(tmp_path / "a.txt"), str(dst_dir))
    assert "перезапись запрещена" in result
    assert (tmp_path / "a.txt").exists()
    assert (dst_dir / "a.txt").read_text(encoding="utf-8") == "старый"


async def test_move_outside_roots_refused(db: Database, tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    (root / "a.txt").write_text("x", encoding="utf-8")
    service = make_service(db, root, execute=True)

    result = await service.move(str(root / "a.txt"), str(tmp_path / "чужое"))
    assert "вне разрешённых" in result
    assert (root / "a.txt").exists()


async def test_move_dir_into_itself_refused(db: Database, tmp_path: Path) -> None:
    folder = tmp_path / "папка"
    folder.mkdir()
    service = make_service(db, tmp_path, execute=True)
    result = await service.move(str(folder), str(folder / "внутрь"))
    assert "внутрь самой себя" in result


async def test_copy_directory_refused(db: Database, tmp_path: Path) -> None:
    folder = tmp_path / "папка"
    folder.mkdir()
    service = make_service(db, tmp_path)
    result = await service.copy(str(folder), str(tmp_path / "куда"))
    assert "копирование папок не поддерживается" in result


async def test_rename_with_path_separator_refused(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path)
    result = await service.rename(str(tmp_path / "a.txt"), "под/именем.txt")
    assert "недопустимое новое имя" in result


async def test_archive_over_limit_refused(db: Database, tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path, max_batch_files=2)
    result = await service.archive(str(tmp_path))
    assert "больше предела" in result


async def test_archive_older_than_filter(db: Database, tmp_path: Path) -> None:
    import os
    import time

    old = tmp_path / "старый.txt"
    old.write_text("x", encoding="utf-8")
    week_ago = time.time() - 8 * 86400
    os.utime(old, (week_ago, week_ago))
    (tmp_path / "новый.txt").write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path)

    result = await service.archive(str(tmp_path), older_than_days=7)
    assert "старый.txt" in result
    assert "новый.txt" not in result


# ── /fileops ─────────────────────────────────────────────────────────────────


async def test_overview_shows_mode_and_batches(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    service = make_service(db, tmp_path)
    text = await service.overview_text()
    assert "сухой прогон" in text
    assert "Журнал пуст" in text

    await service.move(str(tmp_path / "a.txt"), str(tmp_path / "куда"))
    text = await service.overview_text()
    assert "план" in text
    assert "переместить" in text

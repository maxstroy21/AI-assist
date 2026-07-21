"""Инструменты File Ops: регистрация, уровни риска, тексты для модели."""

from pathlib import Path

import pytest

from sba.core.tools.spec import RiskLevel
from sba.infra.config import FileOpsConfig
from sba.infra.db import Database
from sba.modules.files.ops import FileOpsService
from sba.modules.files.opsstore import FileOpsStore
from sba.modules.files.safety import RootGuard
from sba.modules.files.tools import ScanArgs, build_fileops_tools


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_service(db: Database, root: Path, execute: bool = False) -> FileOpsService:
    return FileOpsService(
        FileOpsStore(db), RootGuard([root]), FileOpsConfig(execute=execute)
    )


async def test_seven_tools_registered(db: Database, tmp_path: Path) -> None:
    specs = build_fileops_tools(make_service(db, tmp_path))
    assert {s.name for s in specs} == {
        "move_file", "copy_file", "rename_file", "archive_files",
        "find_duplicates", "find_old_versions", "undo_file_operation",
    }
    assert all(s.module == "files" for s in specs)


async def test_archive_risk_depends_on_mode(db: Database, tmp_path: Path) -> None:
    by_name = {s.name: s for s in build_fileops_tools(make_service(db, tmp_path))}
    # сухой прогон: план безвреден, подтверждение не нужно
    assert by_name["archive_files"].risk == RiskLevel.WRITE
    assert "сухого прогона" in by_name["move_file"].description

    by_name = {
        s.name: s for s in build_fileops_tools(make_service(db, tmp_path, execute=True))
    }
    # боевой режим: массовая операция — destructive (подтверждение)
    assert by_name["archive_files"].risk == RiskLevel.DESTRUCTIVE
    assert by_name["move_file"].risk == RiskLevel.WRITE


async def test_find_duplicates_tool_text(db: Database, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("двойник", encoding="utf-8")
    (tmp_path / "b.txt").write_text("двойник", encoding="utf-8")
    service = make_service(db, tmp_path)
    text = await service.duplicates_text("")
    assert "групп дубликатов: 1" in text
    assert "a.txt" in text and "b.txt" in text


async def test_find_duplicates_scoped_to_folder(db: Database, tmp_path: Path) -> None:
    sub = tmp_path / "внутри"
    sub.mkdir()
    (tmp_path / "a.txt").write_text("двойник", encoding="utf-8")
    (tmp_path / "b.txt").write_text("двойник", encoding="utf-8")
    service = make_service(db, tmp_path)
    text = await service.duplicates_text(str(sub))
    assert "не найдено" in text


async def test_scan_args_default_empty() -> None:
    assert ScanArgs().path == ""


async def test_old_versions_tool_text(db: Database, tmp_path: Path) -> None:
    (tmp_path / "отчёт.docx").write_text("v1", encoding="utf-8")
    (tmp_path / "отчёт (1).docx").write_text("v2", encoding="utf-8")
    service = make_service(db, tmp_path)
    text = await service.old_versions_text("")
    assert "групп похожих версий: 1" in text
    assert "актуальная" in text

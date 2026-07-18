from pathlib import Path

import pytest

import sba.modules.files.tools as files_tools
from sba.infra.config import FilesConfig
from sba.modules.files.tools import FilesToolset, ListArgs, PathArgs


def make_toolset(tmp_path: Path, **overrides) -> FilesToolset:
    config = FilesConfig(
        allowed_roots=[tmp_path], max_list_entries=5, max_read_chars=50, **overrides
    )
    return FilesToolset(config)


async def test_list_files(tmp_path: Path) -> None:
    (tmp_path / "отчёт.txt").write_text("данные", encoding="utf-8")
    (tmp_path / "папка").mkdir()
    result = await make_toolset(tmp_path).list_files(ListArgs(path=str(tmp_path)))
    assert "отчёт.txt" in result
    assert "папка" in result


async def test_relative_path_resolved_against_root(tmp_path: Path) -> None:
    sub = tmp_path / "docs"
    sub.mkdir()
    (sub / "a.md").write_text("x", encoding="utf-8")
    result = await make_toolset(tmp_path).list_files(ListArgs(path="docs"))
    assert "a.md" in result


async def test_path_outside_roots_rejected(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    toolset = make_toolset(root)
    with pytest.raises(ValueError, match="вне разрешённых"):
        toolset._resolve(str(tmp_path / "secret.txt"))


async def test_traversal_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    toolset = make_toolset(root)
    with pytest.raises(ValueError, match="вне разрешённых"):
        toolset._resolve(str(root / ".." / "secret.txt"))


async def test_no_roots_gives_hint(tmp_path: Path) -> None:
    toolset = FilesToolset(FilesConfig())
    with pytest.raises(ValueError, match="allowed_roots"):
        toolset._resolve("что-угодно")


async def test_read_document_truncates(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("а" * 200, encoding="utf-8")
    result = await make_toolset(tmp_path).read_document(PathArgs(path="big.txt"))
    assert "первые 50" in result


async def test_read_binary_refused(tmp_path: Path) -> None:
    (tmp_path / "app.bin").write_bytes(b"\x00\x01\x02data")
    result = await make_toolset(tmp_path).read_document(PathArgs(path="app.bin"))
    assert "не текстовый" in result


async def test_delete_uses_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "старый.txt"
    target.write_text("x", encoding="utf-8")
    trashed: list[str] = []
    monkeypatch.setattr(files_tools, "send2trash", trashed.append)

    result = await make_toolset(tmp_path).delete_file(PathArgs(path="старый.txt"))
    assert trashed == [str(target)]
    assert "корзину" in result

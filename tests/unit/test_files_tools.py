from pathlib import Path

import pytest

import sba.modules.files.tools as files_tools
from sba.infra.config import FilesConfig
from sba.modules.files.tools import FilesToolset, FindArgs, ListArgs, PathArgs


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


async def test_root_referenced_by_its_name(tmp_path: Path) -> None:
    # «покажи папку Downloads» → имя корня должно означать сам корень
    root = tmp_path / "Downloads"
    root.mkdir()
    (root / "файл.txt").write_text("x", encoding="utf-8")
    result = await make_toolset(root).list_files(ListArgs(path="downloads"))
    assert "файл.txt" in result


async def test_dot_lists_allowed_roots(tmp_path: Path) -> None:
    result = await make_toolset(tmp_path).list_files(ListArgs(path="."))
    assert "Разрешённые папки" in result
    assert str(tmp_path) in result


async def test_nonexistent_folder_mentions_allowed_roots(tmp_path: Path) -> None:
    result = await make_toolset(tmp_path).list_files(ListArgs(path="нет-такой"))
    assert "не существует" in result
    assert str(tmp_path) in result


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


async def test_find_files_recursive(tmp_path: Path) -> None:
    nested = tmp_path / "проекты" / "логи"
    nested.mkdir(parents=True)
    (nested / "debug.log").write_text("лог", encoding="utf-8")
    result = await make_toolset(tmp_path).find_files(FindArgs(name_pattern="debug.log"))
    assert str(nested / "debug.log") in result


async def test_find_files_mask(tmp_path: Path) -> None:
    (tmp_path / "a.log").write_text("x", encoding="utf-8")
    (tmp_path / "b.log").write_text("x", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    result = await make_toolset(tmp_path).find_files(FindArgs(name_pattern="*.log"))
    assert "a.log" in result and "b.log" in result
    assert "c.txt" not in result


async def test_find_files_nothing(tmp_path: Path) -> None:
    result = await make_toolset(tmp_path).find_files(FindArgs(name_pattern="ghost.md"))
    assert "Ничего не найдено" in result


async def test_read_bare_name_auto_found_in_subfolder(tmp_path: Path) -> None:
    nested = tmp_path / "глубоко" / "внутри"
    nested.mkdir(parents=True)
    (nested / "заметка.md").write_text("важный текст", encoding="utf-8")
    result = await make_toolset(tmp_path).read_document(PathArgs(path="заметка.md"))
    assert "важный текст" in result
    assert str(nested / "заметка.md") in result  # видно, какой файл прочитан


async def test_read_bare_name_multiple_matches_asks_to_clarify(tmp_path: Path) -> None:
    for sub in ("один", "два"):
        folder = tmp_path / sub
        folder.mkdir()
        (folder / "отчёт.txt").write_text("x", encoding="utf-8")
    result = await make_toolset(tmp_path).read_document(PathArgs(path="отчёт.txt"))
    assert "несколько" in result
    assert "один" in result and "два" in result


async def test_delete_uses_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "старый.txt"
    target.write_text("x", encoding="utf-8")
    trashed: list[str] = []
    monkeypatch.setattr(files_tools, "send2trash", trashed.append)

    result = await make_toolset(tmp_path).delete_file(PathArgs(path="старый.txt"))
    assert trashed == [str(target)]
    assert "корзину" in result

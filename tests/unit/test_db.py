"""База данных: миграции и настройки соединения (Sprint 9)."""

from __future__ import annotations

from pathlib import Path

from sba.infra.db import Database


async def test_busy_timeout_set(tmp_path: Path) -> None:
    """Базу делят бот и MCP-сервер — соединение должно ждать блокировку,
    а не падать сразу с 'database is locked'."""
    db = await Database.open(tmp_path / "sba.db")
    try:
        row = await db.fetch_one("PRAGMA busy_timeout")
        assert row is not None
        assert int(row[0]) == 5000
    finally:
        await db.close()


async def test_wal_mode_enabled(tmp_path: Path) -> None:
    db = await Database.open(tmp_path / "sba.db")
    try:
        row = await db.fetch_one("PRAGMA journal_mode")
        assert row is not None
        assert str(row[0]).lower() == "wal"
    finally:
        await db.close()

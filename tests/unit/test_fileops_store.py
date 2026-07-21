"""Undo-журнал файловых операций: батчи, записи, выбор последнего отменяемого."""

from pathlib import Path

import pytest

from sba.infra.db import Database
from sba.modules.files.opsstore import FileOpsStore


@pytest.fixture
async def store(tmp_path: Path) -> FileOpsStore:
    database = await Database.open(tmp_path / "test.db")
    yield FileOpsStore(database)
    await database.close()


async def test_last_undoable_skips_plans_and_undo_batches(store: FileOpsStore) -> None:
    real = await store.create_batch(
        "move", dry_run=False, description="реальный", entries=[("move", "a", "b")]
    )
    await store.mark_entry(real, 0)
    await store.create_batch(
        "move", dry_run=True, description="план", entries=[("move", "c", "d")]
    )
    await store.create_batch(
        "undo", dry_run=False, description="отмена", entries=[("move", "b", "a")],
        undo_of=real,
    )
    last = await store.last_undoable()
    assert last is not None
    assert last.id == real


async def test_mark_undone_removes_from_undoable(store: FileOpsStore) -> None:
    batch = await store.create_batch(
        "move", dry_run=False, description="x", entries=[("move", "a", "b")]
    )
    await store.mark_undone(batch)
    assert await store.last_undoable() is None


async def test_entries_keep_order_and_errors(store: FileOpsStore) -> None:
    batch = await store.create_batch(
        "archive", dry_run=False, description="x",
        entries=[("move", "a", "b"), ("move", "c", "d")],
    )
    await store.mark_entry(batch, 0)
    await store.mark_entry(batch, 1, error="занято")
    entries = await store.entries(batch)
    assert [e.seq for e in entries] == [0, 1]
    assert entries[0].executed and entries[0].error is None
    assert not entries[1].executed and entries[1].error == "занято"

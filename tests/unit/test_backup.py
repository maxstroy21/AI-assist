"""Бэкап с restore-тестом (Sprint 10): снимок, архив, проверка, ротация."""

import sqlite3
import time as time_mod
import zipfile
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from sba.core.events import EventBus
from sba.infra.db import Database
from sba.modules.backup.service import (
    BACKUP_JOB_ID,
    BACKUP_TOPIC,
    BackupError,
    BackupService,
)
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.scheduler.store import SchedulerStore

TZ = ZoneInfo("Europe/Moscow")


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "data" / "sba.db")
    yield database
    await database.close()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "default.yaml").write_text("app: {}\n", encoding="utf-8")
    (cfg / "local.yaml").write_text("channels: {}\n", encoding="utf-8")
    (cfg / "models.yaml").write_text("models: {}\n", encoding="utf-8")
    return cfg


def make(
    db: Database,
    tmp_path: Path,
    config_dir: Path,
    scheduler: SchedulerService | None = None,
    keep_last: int = 14,
    timeout_seconds: float = 30.0,
) -> BackupService:
    return BackupService(
        db,
        scheduler,
        TZ,
        data_dir=tmp_path / "data",
        config_dir=config_dir,
        at=time(14, 0),
        keep_last=keep_last,
        timeout_seconds=timeout_seconds,
    )


async def test_backup_creates_verified_zip(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    # в базе есть данные — факт переживёт цикл бэкапа
    now = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO memory_facts"
        " (id, user_id, type, subject, content, source, created_at, updated_at)"
        " VALUES ('f1', 'owner', 'fact', 'тест', 'бэкап работает', 'explicit', ?, ?)",
        (now, now),
    )
    service = make(db, tmp_path, config_dir)

    result = await service.run_backup()

    assert result.path.is_file()
    assert result.size_bytes > 0
    assert result.tables > 5           # все таблицы схемы на месте
    with zipfile.ZipFile(result.path) as zf:
        names = set(zf.namelist())
    assert "sba.db" in names
    assert "config/local.yaml" in names
    assert "config/models.yaml" in names
    # снимок из архива — рабочая база: данные читаются обычным sqlite
    extract_dir = tmp_path / "restore"
    with zipfile.ZipFile(result.path) as zf:
        zf.extract("sba.db", extract_dir)
    conn = sqlite3.connect(extract_dir / "sba.db")
    try:
        row = conn.execute("SELECT content FROM memory_facts WHERE id='f1'").fetchone()
    finally:
        conn.close()
    assert row == ("бэкап работает",)
    # временные файлы за собой убраны
    leftovers = [p.name for p in (tmp_path / "data" / "backups").iterdir()]
    assert leftovers == [result.path.name]


async def test_missing_config_files_are_skipped(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    (config_dir / "local.yaml").unlink()  # local.yaml нет — не падаем
    service = make(db, tmp_path, config_dir)
    result = await service.run_backup()
    with zipfile.ZipFile(result.path) as zf:
        names = set(zf.namelist())
    assert "config/local.yaml" not in names
    assert "config/models.yaml" in names


async def test_verify_rejects_corrupted_archive(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    service = make(db, tmp_path, config_dir)
    backups = tmp_path / "data" / "backups"
    backups.mkdir(parents=True)
    bad = backups / "sba-backup-20260101-000000.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("sba.db", b"this is not a database at all")
    with pytest.raises(BackupError):
        service._verify_zip(bad, expected_version=1, expected_tables={"facts"})


async def test_run_backup_deletes_archive_that_fails_verify(
    db: Database, tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make(db, tmp_path, config_dir)

    def broken_verify(
        archive: Path, expected_version: int, expected_tables: set[str]
    ) -> int:
        raise BackupError("имитация битой копии")

    monkeypatch.setattr(service, "_verify_zip", broken_verify)
    with pytest.raises(BackupError):
        await service.run_backup()
    assert list((tmp_path / "data" / "backups").glob("*.zip")) == []


async def test_rotation_keeps_only_last_n(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    backups = tmp_path / "data" / "backups"
    backups.mkdir(parents=True)
    for stamp in ("20250101-000000", "20250102-000000", "20250103-000000"):
        (backups / f"sba-backup-{stamp}.zip").write_bytes(b"old")
    service = make(db, tmp_path, config_dir, keep_last=2)

    result = await service.run_backup()

    names = sorted(p.name for p in backups.glob("*.zip"))
    assert len(names) == 2
    assert result.path.name in names          # свежий остался
    assert "sba-backup-20250101-000000.zip" not in names  # старейшие ушли


async def test_timeout_is_enforced(
    db: Database, tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make(db, tmp_path, config_dir, timeout_seconds=0.05)

    def slow_archive(*args: object, **kwargs: object) -> None:
        time_mod.sleep(1.0)

    monkeypatch.setattr(service, "_archive_and_verify", slow_archive)
    with pytest.raises(BackupError, match="не уложился"):
        await service.run_backup()


async def test_on_job_fired_ignores_foreign_topics(
    db: Database, tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make(db, tmp_path, config_dir)
    calls: list[str] = []

    async def fake_backup() -> None:
        calls.append("run")

    monkeypatch.setattr(service, "run_backup", fake_backup)
    foreign = JobFired(
        job_id="x", topic="reminder.fire", payload={}, scheduled_for=datetime.now(UTC)
    )
    await service.on_job_fired(foreign)
    assert calls == []
    ours = JobFired(
        job_id=BACKUP_JOB_ID, topic=BACKUP_TOPIC, payload={},
        scheduled_for=datetime.now(UTC),
    )
    await service.on_job_fired(ours)
    assert calls == ["run"]


async def test_on_job_fired_survives_backup_failure(
    db: Database, tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = make(db, tmp_path, config_dir)

    async def failing_backup() -> None:
        raise BackupError("диск переполнен")

    monkeypatch.setattr(service, "run_backup", failing_backup)
    event = JobFired(
        job_id=BACKUP_JOB_ID, topic=BACKUP_TOPIC, payload={},
        scheduled_for=datetime.now(UTC),
    )
    await service.on_job_fired(event)   # не бросает — джоб не роняет приложение
    assert "диск переполнен" in service.last_status


async def test_schedule_registers_daily_job_idempotently(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    bus = EventBus()
    scheduler = SchedulerService(SchedulerStore(db), bus, TZ)
    service = make(db, tmp_path, config_dir, scheduler=scheduler)

    await service.schedule()
    job = await scheduler.get(BACKUP_JOB_ID)
    assert job is not None
    assert job.rrule == "FREQ=DAILY"
    first_fire = job.next_fire_at

    # повторная регистрация (рестарт) не сдвигает созревшее срабатывание
    await service.schedule()
    job2 = await scheduler.get(BACKUP_JOB_ID)
    assert job2 is not None
    assert job2.next_fire_at == first_fire


async def test_scheduler_fires_backup_end_to_end(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    """Джоб созрел → JobFired → бэкап реально создан (вся цепочка)."""
    bus = EventBus()
    scheduler = SchedulerService(SchedulerStore(db), bus, TZ)
    service = make(db, tmp_path, config_dir, scheduler=scheduler)
    bus.subscribe(JobFired, service.on_job_fired)

    await scheduler.schedule_once(
        BACKUP_TOPIC, {}, datetime.now(UTC) - timedelta(minutes=1),
        job_id=BACKUP_JOB_ID,
    )
    assert await scheduler.tick() == 1
    archives = list((tmp_path / "data" / "backups").glob("sba-backup-*.zip"))
    assert len(archives) == 1
    assert "✅" in service.last_status


async def test_overview_text_runs_backup_and_lists(
    db: Database, tmp_path: Path, config_dir: Path
) -> None:
    service = make(db, tmp_path, config_dir)
    text = await service.overview_text()
    assert "💾 Бэкап создан" in text
    assert "Проверка восстановления пройдена" in text
    assert "sba-backup-" in text

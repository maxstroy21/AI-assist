"""Бэкап SQLite + конфигов с автоматическим restore-тестом (Sprint 10, шаг 1).

Что бэкапим: data/sba.db — единственный источник истины (задачи, память,
эпизоды, напоминания, аудит, тексты фрагментов документов) плюс YAML-конфиги
(local.yaml с токеном, models.yaml). Векторный индекс Qdrant НЕ бэкапим:
он полностью восстанавливается переиндексацией из SQLite
(`python scripts/reindex.py --full`), а embedded-Qdrant нельзя безопасно
копировать, пока приложение работает.

Как: `VACUUM INTO` даёт атомарный консистентный снимок работающей базы
(WAL не мешает), снимок и конфиги упаковываются в zip. Сразу после создания —
restore-тест: база ЧИТАЕТСЯ ОБРАТНО ИЗ ZIP (проверяется сам архив, а не
промежуточный файл), `PRAGMA integrity_check`, совпадение версии схемы и
набора таблиц с живой базой. Битый архив удаляется — «есть бэкап» значит
«из него можно восстановиться». Ротация: храним последние keep_last копий.

Расписание — ежедневный джоб через общий Scheduler (misfire=deliver: ноутбук
владельца ночью выключен, пропущенный бэкап догоняет при следующем запуске).
Команда /backup — бэкап по требованию мимо LLM.
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, time, tzinfo
from pathlib import Path

import structlog

from sba.infra.db import Database
from sba.modules.scheduler.service import JobFired, SchedulerService

log = structlog.get_logger(__name__)

BACKUP_TOPIC = "backup.daily"
BACKUP_JOB_ID = "backup_daily"

CONFIG_FILES = ("default.yaml", "local.yaml", "models.yaml")
SNAPSHOT_NAME = "sba.db"          # имя базы внутри архива
_TMP_SNAPSHOT = ".snapshot.tmp.db"    # временный снимок до упаковки
_TMP_VERIFY = ".verify.tmp.db"        # временная распаковка для restore-теста
_PREFIX = "sba-backup-"


class BackupError(Exception):
    pass


@dataclass(frozen=True)
class BackupResult:
    path: Path
    size_bytes: int
    tables: int
    schema_version: int


class BackupService:
    def __init__(
        self,
        db: Database,
        scheduler: SchedulerService | None,
        timezone: tzinfo,
        data_dir: Path,
        config_dir: Path,
        at: time,
        keep_last: int,
        timeout_seconds: float,
        backups_dir: Path | None = None,
    ) -> None:
        self._db = db
        self._scheduler = scheduler
        self._tz = timezone
        self._data_dir = data_dir
        self._config_dir = config_dir
        self._at = at
        self._keep_last = keep_last
        self._timeout = timeout_seconds
        self._dir = backups_dir if backups_dir is not None else data_dir / "backups"
        self._last_status = ""   # для /backup: результат последнего запуска

    # ── расписание ───────────────────────────────────────────────────────────

    async def schedule(self) -> None:
        """Идемпотентная регистрация ежедневного джоба (как у утренней сводки):
        существующий джоб с тем же временем не переписывается, иначе рестарт
        сдвинул бы созревшее срабатывание на завтра."""
        if self._scheduler is None:
            return
        existing = await self._scheduler.get(BACKUP_JOB_ID)
        if (
            existing is not None
            and existing.rrule == "FREQ=DAILY"
            and existing.dtstart is not None
            and (existing.dtstart.hour, existing.dtstart.minute)
            == (self._at.hour, self._at.minute)
        ):
            return
        dtstart = datetime.now(self._tz).replace(
            hour=self._at.hour, minute=self._at.minute, second=0, microsecond=0
        )
        await self._scheduler.schedule_rrule(
            BACKUP_TOPIC,
            {},
            rrule="FREQ=DAILY",
            dtstart=dtstart,
            job_id=BACKUP_JOB_ID,
            misfire="deliver",   # пропущенный бэкап догоняет при следующем запуске
        )
        log.info("backup_scheduled", at=self._at.isoformat("minutes"))

    async def on_job_fired(self, event: JobFired) -> None:
        if event.topic != BACKUP_TOPIC:
            return
        try:
            await self.run_backup()
        except Exception as exc:  # джоб не должен ронять приложение
            log.error("backup_failed", error=str(exc))
            self._last_status = f"⚠️ Бэкап не создан: {exc}"

    # ── бэкап ────────────────────────────────────────────────────────────────

    async def run_backup(self) -> BackupResult:
        """Снимок → zip → restore-тест → ротация. BackupError — не получилось."""
        self._dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._dir / _TMP_SNAPSHOT
        stamp = datetime.now(self._tz).strftime("%Y%m%d-%H%M%S")
        target = self._dir / f"{_PREFIX}{stamp}.zip"
        try:
            # атомарный консистентный снимок работающей базы (WAL не мешает);
            # VACUUM INTO отказывается писать в существующий файл — чистим
            snapshot.unlink(missing_ok=True)
            await self._db.execute("VACUUM INTO ?", (str(snapshot),))

            expected_version, expected_tables = await self._live_schema()

            # упаковка и restore-тест — синхронные и дисковые: в worker-поток
            # и под таймаутом (урок «всё внешнее обязано иметь таймаут»)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._archive_and_verify,
                    snapshot,
                    target,
                    expected_version,
                    expected_tables,
                ),
                timeout=self._timeout,
            )
        except TimeoutError:
            target.unlink(missing_ok=True)
            raise BackupError(
                f"бэкап не уложился в {self._timeout:.0f} с — диск занят?"
            ) from None
        except BackupError:
            raise
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise BackupError(str(exc)) from exc
        finally:
            snapshot.unlink(missing_ok=True)

        removed = self._rotate()
        log.info(
            "backup_done",
            path=str(result.path),
            size_bytes=result.size_bytes,
            tables=result.tables,
            rotated_out=removed,
        )
        self._last_status = (
            f"✅ {result.path.name} ({_human_size(result.size_bytes)}), "
            f"проверка восстановления пройдена"
        )
        return result

    async def _live_schema(self) -> tuple[int, set[str]]:
        """Версия схемы и набор таблиц живой базы — эталон для restore-теста."""
        row = await self._db.fetch_one("PRAGMA user_version")
        version = int(row[0]) if row is not None else 0
        rows = await self._db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return version, {r["name"] for r in rows}

    def _archive_and_verify(
        self,
        snapshot: Path,
        target: Path,
        expected_version: int,
        expected_tables: set[str],
    ) -> BackupResult:
        """Синхронная часть (worker-поток): zip + чтение обратно из архива."""
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(snapshot, SNAPSHOT_NAME)
            for name in CONFIG_FILES:
                source = self._config_dir / name
                if source.is_file():
                    zf.write(source, f"config/{name}")
        try:
            tables = self._verify_zip(target, expected_version, expected_tables)
        except BackupError:
            # битый архив не должен притворяться бэкапом
            target.unlink(missing_ok=True)
            raise
        return BackupResult(
            path=target,
            size_bytes=target.stat().st_size,
            tables=tables,
            schema_version=expected_version,
        )

    def _verify_zip(
        self, archive: Path, expected_version: int, expected_tables: set[str]
    ) -> int:
        """Restore-тест: база из архива открывается и совпадает с живой.

        Возвращает число таблиц. BackupError — восстановиться не получится."""
        verify_db = self._dir / _TMP_VERIFY
        try:
            with zipfile.ZipFile(archive) as zf:
                if SNAPSHOT_NAME not in zf.namelist():
                    raise BackupError("в архиве нет sba.db")
                with zf.open(SNAPSHOT_NAME) as src, open(verify_db, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            conn = sqlite3.connect(verify_db)
            try:
                check = conn.execute("PRAGMA integrity_check").fetchone()
                if check is None or check[0] != "ok":
                    raise BackupError(
                        f"integrity_check: {check[0] if check else 'нет ответа'}"
                    )
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version != expected_version:
                    raise BackupError(
                        f"версия схемы {version} != живой {expected_version}"
                    )
                names = {
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                missing = expected_tables - names
                if missing:
                    raise BackupError(f"в копии нет таблиц: {', '.join(sorted(missing))}")
            finally:
                conn.close()
        except BackupError:
            raise
        except Exception as exc:  # zip битый, файл не открылся и т.п.
            raise BackupError(f"restore-тест не прошёл: {exc}") from exc
        finally:
            verify_db.unlink(missing_ok=True)
        return len(names)

    def _rotate(self) -> int:
        """Удалить копии сверх keep_last (имена содержат метку времени —
        лексикографический порядок совпадает с хронологическим)."""
        archives = sorted(self._dir.glob(f"{_PREFIX}*.zip"), reverse=True)
        removed = 0
        for stale in archives[self._keep_last :]:
            stale.unlink(missing_ok=True)
            removed += 1
        return removed

    # ── команда /backup ──────────────────────────────────────────────────────

    async def overview_text(self) -> str:
        """Бэкап по требованию + список копий (команда /backup, мимо LLM)."""
        try:
            result = await self.run_backup()
            head = (
                f"💾 Бэкап создан: {result.path.name} "
                f"({_human_size(result.size_bytes)})\n"
                f"✅ Проверка восстановления пройдена: целостность, "
                f"версия схемы {result.schema_version}, таблиц: {result.tables}"
            )
        except BackupError as exc:
            head = f"⚠️ Бэкап не создан: {exc}"
        archives = sorted(self._dir.glob(f"{_PREFIX}*.zip"), reverse=True)
        lines = [head, f"Папка: {self._dir}"]
        lines.append(f"Хранится копий: {len(archives)} (лимит {self._keep_last})")
        for archive in archives[:5]:
            lines.append(f"• {archive.name} ({_human_size(archive.stat().st_size)})")
        if len(archives) > 5:
            lines.append(f"…и ещё {len(archives) - 5}")
        lines.append("Восстановление: README, раздел «Бэкап и восстановление».")
        return "\n".join(lines)

    @property
    def last_status(self) -> str:
        return self._last_status


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} Б"

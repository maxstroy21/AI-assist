"""Индексатор: периодический скан источников + фоновый воркер очереди.

Отступление от плана (watchdog): наблюдение — периодическим полным сканом
(по умолчанию раз в 2 минуты) — DoD «новый файл находится в течение минут»
закрывается без моста «FS-поток → asyncio». watchdog можно добавить позже
как ускоритель, скан всё равно обязателен (диффы после простоя/рестарта).

Индексация не мешает диалогу: перед тяжёлыми шагами воркер ждёт паузы
в общении (cooldown после последнего сообщения) — эмбеддинг-запросы
не конкурируют с генерацией ответа на CPU.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from sba.infra.config import RagConfig
from sba.llm.gateway import LLMError
from sba.modules.indexer.chunker import make_chunks
from sba.modules.indexer.extractors import ExtractError, extract
from sba.modules.indexer.store import CatalogStore
from sba.modules.rag.interface import DocumentIndex

log = structlog.get_logger(__name__)

EXTRACT_TIMEOUT_SECONDS = 120.0  # всё внешнее обязано иметь таймаут
# потолок на эмбеддинг одного файла: bge-m3 на CPU медленна, а при локальной
# чат-модели ещё и конкурирует с ней за память — один файл не должен вешать
# индексатор надолго (иначе health-монитор считает его зависшим). Превысили —
# трактуем как временную недоступность модели: файл остаётся в очереди
INDEX_EMBED_TIMEOUT_SECONDS = 300.0
MAX_ATTEMPTS = 3
QUEUE_BATCH = 20


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class IndexerService:
    def __init__(
        self,
        catalog: CatalogStore,
        index: DocumentIndex,
        config: RagConfig,
    ) -> None:
        self._catalog = catalog
        self._index = index
        self._config = config
        self._extensions = {ext.lower() for ext in config.include_extensions}
        self._last_activity = 0.0
        self._last_scan_at: float | None = None
        self._heartbeat: Callable[[], None] | None = None

    # ── связь с диалогом ─────────────────────────────────────────────────────

    def notice_activity(self) -> None:
        """Вызывается на каждое входящее сообщение (подписка на шину в app.py)."""
        self._last_activity = time.monotonic()

    def _beat(self) -> None:
        """Сигнал жизни health-монитору. Бьём по ходу обработки (каждый файл,
        каждый тик ожидания), а не раз за цикл: разбор большого корпуса и
        намеренная пауза под диалог — это работа, а не зависание."""
        if self._heartbeat is not None:
            self._heartbeat()

    async def _wait_quiet(self) -> None:
        while True:
            self._beat()  # ждём паузы в диалоге — это не простой, монитор жив
            since = time.monotonic() - self._last_activity
            remaining = self._config.dialog_cooldown_seconds - since
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 5.0))

    # ── скан источников ──────────────────────────────────────────────────────

    def _iter_source_files(self) -> list[Path]:
        found: list[Path] = []
        max_bytes = self._config.max_file_mb * 1024 * 1024
        for source in self._config.sources:
            root = Path(source).expanduser()
            if not root.is_dir():
                log.warning("rag_source_missing", source=str(root))
                continue
            for path in root.rglob("*"):
                # скрытые каталоги (.obsidian, .git) не индексируем
                if any(part.startswith(".") for part in path.relative_to(root).parts[:-1]):
                    continue
                # временные lock-файлы Office (Excel/Word): «~$Книга.xlsx» —
                # заблокированы открытым приложением и не несут содержимого
                if path.name.startswith("~$"):
                    continue
                try:
                    if not path.is_file() or path.suffix.lower() not in self._extensions:
                        continue
                    if path.stat().st_size > max_bytes:
                        log.warning("rag_file_too_large", path=str(path))
                        continue
                except OSError as exc:
                    # заблокированный/недоступный файл не должен рушить скан
                    log.warning("rag_file_unreadable", path=str(path), error=str(exc))
                    continue
                found.append(path)
        return found

    async def scan_once(self) -> tuple[int, int]:
        """Диффы диска против каталога → очередь. Возвращает (enqueued, deleted)."""
        on_disk = await asyncio.to_thread(self._iter_source_files)
        seen = {str(p) for p in on_disk}
        enqueued = 0
        for path in on_disk:
            self._beat()  # хэширование большого корпуса — работа, не зависание
            try:
                stat = path.stat()
                record = await self._catalog.get(str(path))
                unchanged = (
                    record is not None
                    and record.mtime == stat.st_mtime
                    and record.size == stat.st_size
                )
                if unchanged and record is not None and record.status != "pending":
                    continue
                content_hash = await asyncio.to_thread(_file_hash, path)
            except OSError as exc:
                # файл заблокирован (открыт в Excel) или недоступен: пропускаем
                # этот заход, вернёмся к нему в следующем скане — весь скан не рушим
                log.warning("rag_file_scan_skipped", path=str(path), error=str(exc))
                continue
            if record is not None and record.content_hash == content_hash:
                if record.status == "pending":
                    # прошлый заход не дошёл до индексации — вернём в очередь
                    await self._catalog.enqueue(str(path), "upsert")
                    enqueued += 1
                else:
                    await self._catalog.touch(str(path), stat.st_mtime, stat.st_size)
                continue
            await self._catalog.upsert_pending(
                str(path), content_hash, stat.st_mtime, stat.st_size
            )
            await self._catalog.enqueue(str(path), "upsert")
            enqueued += 1

        deleted = 0
        for record in await self._catalog.all_files():
            if record.path not in seen:
                await self._catalog.enqueue(record.path, "delete")
                deleted += 1
        self._last_scan_at = time.monotonic()
        if enqueued or deleted:
            log.info("rag_scan_done", enqueued=enqueued, deleted=deleted)
        return enqueued, deleted

    # ── воркер очереди ───────────────────────────────────────────────────────

    async def process_queue(self, respect_cooldown: bool = True) -> int:
        """Обрабатывает очередь до опустошения; возвращает число успехов.

        LLMError (эмбеддинг-модель недоступна) прерывает заход без сжигания
        попыток — файл останется в очереди до следующего цикла.
        """
        processed = 0
        while batch := await self._catalog.queue_batch(QUEUE_BATCH):
            progressed = False
            for item in batch:
                self._beat()  # каждый файл — сигнал жизни: длинный разбор ≠ зависание
                if respect_cooldown:
                    await self._wait_quiet()
                try:
                    await self._process_item(item.path, item.op)
                except LLMError as exc:
                    log.warning("rag_index_llm_unavailable", error=str(exc))
                    return processed
                except (ExtractError, TimeoutError, OSError) as exc:
                    log.warning("rag_index_failed", path=item.path, error=str(exc))
                    if item.attempts + 1 >= MAX_ATTEMPTS:
                        await self._catalog.mark_failed(item.path, str(exc))
                        await self._catalog.queue_remove(item.path)
                        progressed = True
                    else:
                        await self._catalog.queue_bump_attempts(item.path)
                    continue
                await self._catalog.queue_remove(item.path)
                processed += 1
                progressed = True
            if not progressed:
                break  # вся пачка ждёт повтора — не крутимся вхолостую
        return processed

    async def _process_item(self, path_str: str, op: str) -> None:
        record = await self._catalog.get(path_str)
        still_exists = await asyncio.to_thread(Path(path_str).is_file)
        if op == "delete" or not still_exists:
            if record is not None:
                await self._index.delete_doc(record.id)
                await self._catalog.remove(path_str)
                log.info("rag_file_removed", path=path_str)
            return
        if record is None:  # файл в очереди, но не в каталоге — не должно случаться
            return
        blocks = await asyncio.wait_for(
            asyncio.to_thread(extract, Path(path_str)), timeout=EXTRACT_TIMEOUT_SECONDS
        )
        chunks = make_chunks(
            blocks, self._config.chunk_chars, self._config.chunk_overlap_chars
        )
        self._beat()  # перед медленным эмбеддингом — сигнал жизни health-монитору
        try:
            count = await asyncio.wait_for(
                self._index.upsert_chunks(record.id, path_str, chunks),
                timeout=INDEX_EMBED_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            # эмбеддинг не уложился (Ollama занята/конкурирует с чат-моделью за
            # CPU) — не вина файла: как временная недоступность модели (LLMError),
            # файл остаётся в очереди, попытки не жжём, индексатор продолжает жить
            raise LLMError(
                f"эмбеддинг {Path(path_str).name} не уложился в "
                f"{INDEX_EMBED_TIMEOUT_SECONDS:.0f} с — модель занята"
            ) from exc
        await self._catalog.mark_indexed(path_str, count)
        log.info("rag_file_indexed", path=path_str, chunks=count)

    # ── фоновый цикл ─────────────────────────────────────────────────────────

    async def run_forever(self, heartbeat: Callable[[], None] | None = None) -> None:
        self._heartbeat = heartbeat  # scan/process бьют по ходу через self._beat()
        interval = self._config.scan_interval_minutes * 60
        while True:
            self._beat()  # сигнал жизни health-монитору (Sprint 10)
            try:
                await self.scan_once()
                await self.process_queue()
            except Exception as exc:  # цикл не должен умирать ни от чего
                log.error("rag_index_cycle_failed", error=str(exc))
            await asyncio.sleep(interval)

    # ── статус для /rag ──────────────────────────────────────────────────────

    async def stats_text(self) -> str:
        counts = await self._catalog.status_counts()
        queue = await self._catalog.queue_size()
        if not self._config.sources:
            return (
                "Индексация не настроена: добавьте папки в modules.rag.sources "
                "в config/local.yaml."
            )
        lines = [
            "Индекс документов:",
            f"• проиндексировано файлов: {counts.get('indexed', 0)}",
            f"• в очереди: {queue}",
        ]
        if counts.get("failed"):
            lines.append(f"• с ошибками: {counts['failed']} (подробности в логе)")
        if self._last_scan_at is not None:
            minutes = (time.monotonic() - self._last_scan_at) / 60
            lines.append(f"• последний скан: {minutes:.0f} мин назад")
        return "\n".join(lines)

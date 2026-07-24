"""Композиция приложения: конфиг → инфраструктура → ядро → каналы.

Единственное место, которому позволено знать обо всех частях системы
(DI вручную, без фреймворков — docs/02-architecture.md ADR-1).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

from sba import __version__
from sba.channels.cli.repl import CliChannel
from sba.channels.telegram.gateway import TelegramChannel
from sba.channels.webchat.gateway import WebChatChannel
from sba.core.agent.orchestrator import AgentOrchestrator
from sba.core.events import EventBus, MessageReceived, SessionClosed
from sba.core.history import HistoryStore
from sba.core.processor import EchoProcessor
from sba.core.router import Router
from sba.core.tools.registry import ToolRegistry
from sba.core.types import ChannelAdapter, MessageProcessor, OutgoingMessage
from sba.infra.audit import AuditLog
from sba.infra.config import Config, ConfigError, load_config
from sba.infra.db import Database
from sba.infra.logging import setup_logging
from sba.infra.vectors import VectorStore
from sba.llm.config import load_models_config
from sba.llm.service import ModelGateway
from sba.mcp.client_hub import MCPClientHub
from sba.modules.backup.service import BackupService
from sba.modules.basic.tools import build_tools as build_basic_tools
from sba.modules.files.ops import FileOpsService
from sba.modules.files.opsstore import FileOpsStore
from sba.modules.files.safety import RootGuard
from sba.modules.files.tools import FilesToolset, build_fileops_tools
from sba.modules.health.service import HealthMonitor
from sba.modules.indexer.service import IndexerService
from sba.modules.indexer.store import CatalogStore
from sba.modules.memory.consolidation import MemoryConsolidator
from sba.modules.memory.episodes import EpisodeStore
from sba.modules.memory.service import MemoryService
from sba.modules.memory.store import MemoryStore
from sba.modules.memory.tools import build_memory_tools
from sba.modules.rag.service import RAGService
from sba.modules.rag.store import ChunkStore
from sba.modules.rag.tools import build_rag_tools
from sba.modules.reminders.brief import MorningBrief, parse_brief_time
from sba.modules.reminders.service import ReminderService
from sba.modules.reminders.store import ReminderStore
from sba.modules.reminders.tools import build_reminder_tools
from sba.modules.scheduler.service import JobFired, SchedulerService
from sba.modules.scheduler.store import SchedulerStore
from sba.modules.tasks.dates import WhenParser
from sba.modules.tasks.events import TaskChanged
from sba.modules.tasks.service import TasksService
from sba.modules.tasks.store import TaskStore
from sba.modules.tasks.tools import build_task_tools

log = structlog.get_logger(__name__)

# сколько раз пытаться прогреть модель на старте: холодная загрузка на CPU
# может не уложиться в один таймаут чтения, но Ollama догрузит модель в фоне,
# и вторая попытка обычно застаёт её уже в памяти. Больше двух не ждём —
# иначе старт растянется на десятки минут, а это сигнал сменить модель на лёгкую
WARMUP_ATTEMPTS = 2


async def keep_warm_loop(gateway: ModelGateway, interval_seconds: float) -> None:
    """Фоновый прогрев: не даёт Ollama выгрузить chat-модель из RAM.

    Пауза идёт первой: стартовый прогрев уже выполнен в App.run() до приёма
    сообщений, повторять его сразу незачем."""
    while True:
        await asyncio.sleep(interval_seconds)
        await gateway.warmup()


class App:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db: Database | None = None
        self.bus = EventBus()
        self.router: Router | None = None
        self.gateway: ModelGateway | None = None
        self.vectors: VectorStore | None = None
        self.indexer: IndexerService | None = None
        self.scheduler: SchedulerService | None = None
        self.reminders: ReminderService | None = None
        self.brief: MorningBrief | None = None
        self.backup: BackupService | None = None
        self.consolidator: MemoryConsolidator | None = None
        self.health: HealthMonitor | None = None
        self.mcp_hub: MCPClientHub | None = None
        self.channels: list[ChannelAdapter] = []
        # heartbeat-функции фоновых компонентов для health-монитора (Sprint 10);
        # заполняется в create(), передаётся в run_forever() в run()
        self._health_beats: dict[str, Callable[[], None]] = {}

    @classmethod
    async def create(cls, config_dir: Path) -> App:
        config = load_config(config_dir)
        setup_logging(config.logging.level, config.logging.format, config.logging.file)
        log.info("app_starting", version=__version__, data_dir=str(config.app.data_dir))

        app = cls(config)
        app.db = await Database.open(config.app.data_dir / "sba.db")

        processor: MessageProcessor
        if config.agent.processor == "echo":
            processor = EchoProcessor()
        else:
            audit = AuditLog(app.db)
            registry = ToolRegistry(audit)
            for spec in build_basic_tools(config.app.timezone):
                registry.register(spec)
            for spec in FilesToolset(config.files).build_tools():
                registry.register(spec)

            # File Ops (Sprint 8): операции с undo-журналом; по умолчанию —
            # сухой прогон (только планы), см. modules.fileops.execute
            fileops: FileOpsService | None = None
            fileops_cfg = config.modules.fileops
            if fileops_cfg.enabled:
                fileops = FileOpsService(
                    FileOpsStore(app.db),
                    RootGuard(config.files.allowed_roots),
                    fileops_cfg,
                )
                if not fileops_cfg.execute:
                    log.info("fileops_dry_run_mode", hint="планы без исполнения")
                for spec in build_fileops_tools(fileops):
                    registry.register(spec)

            app.gateway = ModelGateway(load_models_config(config_dir / "models.yaml"))

            rag_cfg = config.modules.rag
            rag_service: RAGService | None = None
            if rag_cfg.enabled:
                if not app.gateway.has_role("embedding"):
                    log.warning(
                        "rag_no_embedding_role",
                        hint="добавьте роль embedding в config/models.yaml — "
                        "поиск по документам будет только лексическим",
                    )
                # embedded Qdrant — однопроцессный: если файлы уже открыты (чаще
                # всего — второй запущенный экземпляр бота, либо предыдущий не
                # завершился до конца), открытие падает блокировкой. НЕ роняем
                # весь ассистент из-за поиска: понятно объясняем причину и
                # работаем дальше без векторного индекса (поиск по документам —
                # лексический, всё остальное — как обычно). Урок «внешнее не
                # должно ронять систему», живая проверка Sprint 9
                try:
                    app.vectors = VectorStore.open(config.app.data_dir / "qdrant")
                except Exception as exc:
                    log.error("rag_vectors_locked", error=str(exc))
                    print(
                        "⚠️ Поисковый индекс документов занят и не открылся. Скорее "
                        "всего ассистент УЖЕ ЗАПУЩЕН в другом окне — тогда закройте "
                        "это окно и пользуйтесь тем. Если нет — закройте все окна "
                        "бота, подождите несколько секунд и запустите заново.\n"
                        "   Пока продолжаю без поиска по документам (остальное "
                        "работает).",
                        flush=True,
                    )
                if app.vectors is not None:
                    rag_service = RAGService(
                        ChunkStore(app.db),
                        app.vectors,
                        app.gateway,
                        embed_batch=rag_cfg.embed_batch,
                    )
                    for spec in build_rag_tools(
                        rag_service,
                        top_k=rag_cfg.search_top_k,
                        snippet_chars=rag_cfg.snippet_chars,
                        configured=bool(rag_cfg.sources),
                    ):
                        registry.register(spec)
                    app.indexer = IndexerService(CatalogStore(app.db), rag_service, rag_cfg)

            tasks: TasksService | None = None
            tasks_cfg = config.modules.tasks
            reminders_cfg = config.modules.reminders
            tz = ZoneInfo(config.app.timezone)
            parser: WhenParser | None = None
            if tasks_cfg.enabled or reminders_cfg.enabled:
                if not app.gateway.has_role("extraction"):
                    log.warning(
                        "tasks_no_extraction_role",
                        hint="добавьте роль extraction в config/models.yaml — "
                        "сложные формулировки сроков разбираться не будут",
                    )
                parser = WhenParser(
                    app.gateway if app.gateway.has_role("extraction") else None,
                    tz,
                    default_hour=tasks_cfg.default_hour,
                    clarify_confidence=tasks_cfg.clarify_confidence,
                )
            if tasks_cfg.enabled:
                assert parser is not None
                tasks = TasksService(
                    TaskStore(app.db),
                    parser,
                    tz,
                    list_limit=tasks_cfg.list_limit,
                    bus=app.bus,
                )
                for spec in build_task_tools(tasks):
                    registry.register(spec)

            # Scheduler общий: напоминания/сводка (Sprint 6) и бэкап (Sprint 10)
            backup_cfg = config.modules.backup
            if reminders_cfg.enabled or backup_cfg.enabled:
                app.scheduler = SchedulerService(
                    SchedulerStore(app.db),
                    app.bus,
                    tz,
                    tick_seconds=config.modules.scheduler.tick_seconds,
                    misfire_grace_minutes=config.modules.scheduler.misfire_grace_minutes,
                )

            if reminders_cfg.enabled:
                assert parser is not None
                assert app.scheduler is not None
                # доставка через Router (создаётся ниже) — позднее связывание
                async def deliver_notification(out: OutgoingMessage) -> None:
                    assert app.router is not None
                    await app.router.deliver(out)

                app.reminders = ReminderService(
                    ReminderStore(app.db),
                    app.scheduler,
                    parser,
                    tz,
                    deliver=deliver_notification,
                    targets=app._notify_targets(),
                    audit=audit,
                    tasks=tasks,
                    snooze_minutes=reminders_cfg.snooze_minutes,
                    followup_minutes=int(reminders_cfg.followup_hours * 60),
                    default_hour=tasks_cfg.default_hour,
                )
                reminders = app.reminders
                app.bus.subscribe(JobFired, reminders.on_job_fired)
                app.bus.subscribe(TaskChanged, reminders.on_task_changed)
                for spec in build_reminder_tools(reminders):
                    registry.register(spec)
                if reminders_cfg.morning_brief.enabled:
                    app.brief = MorningBrief(
                        app.scheduler,
                        ReminderStore(app.db),
                        tz,
                        deliver=deliver_notification,
                        targets=app._notify_targets(),
                        audit=audit,
                        at=parse_brief_time(reminders_cfg.morning_brief.time),
                        tasks=tasks,
                    )
                    app.bus.subscribe(JobFired, app.brief.on_job_fired)

            # Бэкап с restore-тестом (Sprint 10): ежедневный джоб + /backup
            if backup_cfg.enabled:
                assert app.scheduler is not None
                app.backup = BackupService(
                    app.db,
                    app.scheduler,
                    tz,
                    data_dir=config.app.data_dir,
                    config_dir=config_dir,
                    at=parse_brief_time(backup_cfg.time),
                    keep_last=backup_cfg.keep_last,
                    timeout_seconds=backup_cfg.timeout_seconds,
                    backups_dir=backup_cfg.dir,
                )
                backup = app.backup
                app.bus.subscribe(JobFired, backup.on_job_fired)

            memory: MemoryService | None = None
            memory_cfg = config.modules.memory
            if memory_cfg.enabled:
                # векторный recall памяти включается вместе с RAG (общий Qdrant
                # и эмбеддер); без него память работает на FTS, как в Sprint 3.
                # semantic=false снимает эмбеддинг-модель с горячего пути ради
                # экономии RAM (см. modules.memory.semantic в config).
                # app.vectors is None — Qdrant не открылся (занят): память тоже
                # уходит на FTS, а не падает
                semantic_ok = memory_cfg.semantic and app.vectors is not None
                mem_vectors = app.vectors if semantic_ok else None
                mem_embedder = app.gateway if semantic_ok else None
                memory_store = MemoryStore(app.db, vectors=mem_vectors, embedder=mem_embedder)
                episode_store = EpisodeStore(app.db)
                memory = MemoryService(memory_store, episodes=episode_store)
                if not memory_cfg.semantic:
                    log.info("memory_semantic_disabled", reason="config: FTS-only recall")
                for spec in build_memory_tools(memory):
                    registry.register(spec)

                # автопамять (Sprint 7): закрытые разговоры → эпизод → факты,
                # фоново и только в паузах диалога (CPU не конкурирует с ответами)
                if memory_cfg.auto_extract:
                    if app.gateway.has_role("extraction") and app.gateway.has_role(
                        "summarize"
                    ):
                        app.consolidator = MemoryConsolidator(
                            app.db, memory_store, episode_store, app.gateway, memory_cfg
                        )
                        consolidator = app.consolidator
                        app.bus.subscribe(SessionClosed, consolidator.on_session_closed)

                        async def on_dialog_activity(event: MessageReceived) -> None:
                            consolidator.notice_activity()

                        app.bus.subscribe(MessageReceived, on_dialog_activity)
                    else:
                        log.warning(
                            "memory_auto_extract_disabled",
                            hint="нужны роли extraction и summarize в config/models.yaml",
                        )

            # Health-мониторинг (Sprint 10): фоновые компоненты подают сигнал
            # жизни, монитор сам пишет в TG, если кто-то замолчал. Порог «завис»
            # считаем из интервала самого компонента, чтобы нормальная пауза
            # цикла не была ложной тревогой
            health_cfg = config.modules.health
            if health_cfg.enabled:
                async def deliver_health(out: OutgoingMessage) -> None:
                    assert app.router is not None
                    await app.router.deliver(out)

                app.health = HealthMonitor(
                    deliver=deliver_health,
                    targets=app._notify_targets(),
                    check_interval_seconds=health_cfg.check_interval_seconds,
                )

                def silence(interval_seconds: float) -> float:
                    return max(
                        health_cfg.min_silence_seconds,
                        interval_seconds * health_cfg.grace_multiplier,
                    )

                if app.indexer is not None:
                    app._health_beats["indexer"] = app.health.register(
                        "indexer", "индексатор документов",
                        silence(rag_cfg.scan_interval_minutes * 60),
                    )
                if app.scheduler is not None:
                    app._health_beats["scheduler"] = app.health.register(
                        "scheduler", "планировщик напоминаний",
                        silence(config.modules.scheduler.tick_seconds),
                    )
                if app.consolidator is not None:
                    app._health_beats["consolidator"] = app.health.register(
                        "consolidator", "консолидация памяти",
                        silence(memory_cfg.check_interval_seconds),
                    )

            # MCP Client Hub (Sprint 9): инструменты внешних серверов попадают
            # в общий реестр с риском из конфига; недоступный сервер — warning,
            # не отказ старта
            if config.mcp.servers:
                app.mcp_hub = MCPClientHub(config.mcp.servers)
                await app.mcp_hub.start(registry)

            processor = AgentOrchestrator(
                gateway=app.gateway,
                history=HistoryStore(app.db),
                registry=registry,
                audit=audit,
                config=config.agent,
                timezone=config.app.timezone,
                memory=memory,
                extra_commands=app._build_extra_commands(rag_service, tasks, memory, fileops),
                extra_always_modules=(
                    app.mcp_hub.module_names if app.mcp_hub is not None else None
                ),
            )

            if app.indexer is not None:
                indexer = app.indexer

                async def on_message(event: MessageReceived) -> None:
                    indexer.notice_activity()

                app.bus.subscribe(MessageReceived, on_message)

        app.router = Router(
            db=app.db,
            bus=app.bus,
            processor=processor,
            session_idle_timeout=timedelta(minutes=config.session.idle_timeout_minutes),
        )
        if app.reminders is not None:
            app.router.register_action_handler("rem", app.reminders.handle_action)

        if config.channels.cli.enabled:
            cli = CliChannel(handle_incoming=app.router.handle_incoming)
            app.router.register_channel(cli)
            app.channels.append(cli)

        if config.channels.telegram.enabled:
            if not config.channels.telegram.token:
                raise ConfigError(
                    "channels.telegram.token пуст — добавьте токен от @BotFather "
                    "в config/local.yaml"
                )
            telegram = TelegramChannel(
                token=config.channels.telegram.token,
                allowed_user_ids=config.channels.telegram.allowed_user_ids,
                handle_incoming=app.router.handle_incoming,
                handle_action=app.router.handle_action,
            )
            app.router.register_channel(telegram)
            app.channels.append(telegram)

        if config.channels.web.enabled:
            web_cfg = config.channels.web
            if web_cfg.host not in ("127.0.0.1", "localhost", "::1"):
                log.warning(
                    "web_chat_non_local_host",
                    host=web_cfg.host,
                    hint="аутентификации в web-чате нет — не открывайте его в сеть",
                )
            db = app.db
            history_store = HistoryStore(db)

            async def web_history() -> list[dict[str, str]]:
                # история активного разговора web-канала для новой вкладки
                row = await db.fetch_one(
                    "SELECT id FROM conversations"
                    " WHERE user_id=? AND channel=? AND closed_at IS NULL"
                    " ORDER BY started_at DESC LIMIT 1",
                    ("local", "web"),
                )
                if row is None:
                    return []
                entries = await history_store.recent(row["id"], web_cfg.history_messages)
                return [{"role": e.role, "text": e.content} for e in entries]

            web = WebChatChannel(
                host=web_cfg.host,
                port=web_cfg.port,
                handle_incoming=app.router.handle_incoming,
                handle_action=app.router.handle_action,
                fetch_history=web_history,
            )
            app.router.register_channel(web)
            app.channels.append(web)

        return app

    def _notify_targets(self) -> list[tuple[str, str]]:
        """Каналы доставки уведомлений (напоминания, сводка): все активные."""
        targets: list[tuple[str, str]] = []
        telegram_cfg = self.config.channels.telegram
        if telegram_cfg.enabled and telegram_cfg.allowed_user_ids:
            targets.append(("telegram", str(telegram_cfg.allowed_user_ids[0])))
        if self.config.channels.cli.enabled:
            targets.append(("cli", "local"))
        if self.config.channels.web.enabled:
            targets.append(("web", "local"))
        return targets

    def _build_extra_commands(
        self,
        rag_service: RAGService | None,
        tasks: TasksService | None,
        memory: MemoryService | None = None,
        fileops: FileOpsService | None = None,
    ) -> dict[str, tuple[str, Callable[[], Awaitable[str]]]]:
        """Сервис-команды модулей для оркестратора (инъекция: ядро не знает модулей)."""
        commands: dict[str, tuple[str, Callable[[], Awaitable[str]]]] = {}
        if tasks is not None:
            commands["/tasks"] = ("открытые задачи по срокам", tasks.overview_text)
        if self.reminders is not None:
            commands["/reminders"] = ("предстоящие напоминания", self.reminders.overview_text)
        if memory is not None:
            # ручная ревизия автопамяти (риск Sprint 7): видно, что запомнилось
            commands["/memory"] = ("что запомнено за неделю", memory.review_text)
        if fileops is not None:
            # ревизия сухого прогона (риск Sprint 8): режим и журнал операций
            commands["/fileops"] = ("журнал файловых операций", fileops.overview_text)
        if self.backup is not None:
            # бэкап по требованию + список копий (Sprint 10)
            commands["/backup"] = ("сделать бэкап и показать копии", self.backup.overview_text)
        if self.health is not None:
            # ревизия здоровья фоновых компонентов (Sprint 10)
            commands["/health"] = ("состояние фоновых компонентов", self.health.overview_text)
        if self.indexer is not None and rag_service is not None:
            indexer, rag = self.indexer, rag_service

            async def rag_status() -> str:
                text = await indexer.stats_text()
                chunks = await rag.chunk_count()
                return f"{text}\n• фрагментов в поисковом индексе: {chunks}"

            commands["/rag"] = ("состояние индекса документов", rag_status)
        return commands

    async def run(self) -> None:
        """Работает, пока живы каналы (при одном интерактивном канале —
        выход из REPL завершает приложение; сервис-режим живёт на Telegram)."""
        if not self.channels:
            log.error("no_channels_enabled")
            return
        background: list[asyncio.Task[None]] = []
        if self.gateway is not None and not self.gateway.chat_runtime_is_local():
            # облачная модель: RAM греть нечего и keep-warm не нужен (на
            # бесплатном тарифе он бы ещё и съедал дневной лимит запросов).
            # Но один пробный запрос делаем: ошибка ключа или сети видна
            # сразу на старте, а не на первом вопросе владельца
            log.info("cloud_model_check")
            print("⏳ Проверяю доступ к облачной модели…", flush=True)
            if await self.gateway.warmup():
                print("✅ Модель загружена, можно писать.", flush=True)
                log.info("model_ready")
            else:
                print(
                    "⚠️ Облачная модель не ответила — проверьте интернет и "
                    "API-ключ (переменная окружения из config/models.yaml; "
                    "после setx нужно открыть НОВОЕ окно PowerShell). "
                    "Подробности — строкой выше в логе.",
                    flush=True,
                )
                log.warning("cloud_model_not_ready")
        elif self.gateway is not None and self.config.llm.keep_warm_minutes > 0:
            # прогрев ДО приёма сообщений и ДО старта индексатора: первый вопрос
            # после запуска не ждёт холодную загрузку модели и не ловит ReadTimeout
            # (загрузка на CPU — минуты; в это время бот сознательно молчит)
            log.info("model_warming_up")
            # заметная строка для владельца: среди технических строк лога легко
            # пропустить момент готовности, а до него бот молчит
            print("⏳ Загружаю модель в память, подождите (обычно меньше минуты)…",
                  flush=True)
            # холодная загрузка тяжёлой модели на слабом CPU может превысить таймаут
            # чтения одного запроса — пробуем несколько раз: Ollama продолжает грузить
            # модель в фоне даже после таймаута, поэтому следующая попытка обычно
            # застаёт её уже в памяти
            ready = False
            for attempt in range(WARMUP_ATTEMPTS):
                if await self.gateway.warmup():
                    ready = True
                    break
                if attempt < WARMUP_ATTEMPTS - 1:
                    print("   …модель ещё грузится, продолжаю ждать…", flush=True)
            if ready:
                print("✅ Модель загружена, можно писать.", flush=True)
                log.info("model_ready")
            else:
                # честно: не рапортуем «готово», если модель не поднялась —
                # иначе первый ответ молча падает по таймауту (урок владельца)
                print(
                    "⚠️ Модель пока не загрузилась — на этой машине загрузка идёт "
                    "долго. Можно писать: первый ответ может занять пару минут, а "
                    "если увидите ошибку таймаута — просто повторите сообщение.",
                    flush=True,
                )
                log.warning("model_not_ready_after_warmup")
            background.append(
                asyncio.create_task(
                    keep_warm_loop(self.gateway, self.config.llm.keep_warm_minutes * 60)
                )
            )
        if self.indexer is not None:
            background.append(
                asyncio.create_task(self.indexer.run_forever(self._health_beats.get("indexer")))
            )
        if self.consolidator is not None:
            background.append(
                asyncio.create_task(
                    self.consolidator.run_forever(self._health_beats.get("consolidator"))
                )
            )
        if self.scheduler is not None:
            # порядок важен: сначала регистрация сводки и синхронизация задач,
            # потом цикл — иначе перерегистрация может затереть созревший джоб
            if self.brief is not None:
                await self.brief.schedule()
            if self.backup is not None:
                await self.backup.schedule()
            if self.reminders is not None:
                synced = await self.reminders.sync_open_tasks()
                if synced:
                    log.info("task_reminders_synced", tasks=synced)
            background.append(
                asyncio.create_task(
                    self.scheduler.run_forever(self._health_beats.get("scheduler"))
                )
            )
        if self.health is not None:
            background.append(asyncio.create_task(self.health.run_forever()))
        # Ждём ПЕРВЫЙ завершившийся канал, а не все: выход из консоли (/quit,
        # EOF, Ctrl+C — чтение stdin гасит консольный канал) должен останавливать
        # и Telegram, иначе бот с несколькими каналами не завершается по Ctrl+C
        # (aiogram знай себе опрашивает). Живая проверка 2026-07-24.
        channel_tasks = [asyncio.create_task(ch.start()) for ch in self.channels]
        try:
            done, pending = await asyncio.wait(
                channel_tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    log.error("channel_stopped_with_error", error=str(task.exception()))
        finally:
            for task in background:
                task.cancel()
            await self.shutdown()

    async def shutdown(self) -> None:
        for channel in self.channels:
            await channel.stop()
        if self.mcp_hub is not None:
            await self.mcp_hub.stop()
        if self.gateway is not None:
            await self.gateway.aclose()
        if self.vectors is not None:
            await self.vectors.close()
        if self.db is not None:
            await self.db.close()
        log.info("app_stopped")


async def run_app(config_dir: Path) -> None:
    app = await App.create(config_dir)
    await app.run()

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
from sba.core.agent.orchestrator import AgentOrchestrator
from sba.core.events import EventBus, MessageReceived
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
from sba.modules.basic.tools import build_tools as build_basic_tools
from sba.modules.files.tools import FilesToolset
from sba.modules.indexer.service import IndexerService
from sba.modules.indexer.store import CatalogStore
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
        self.channels: list[ChannelAdapter] = []

    @classmethod
    async def create(cls, config_dir: Path) -> App:
        config = load_config(config_dir)
        setup_logging(config.logging.level, config.logging.format)
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
                app.vectors = VectorStore.open(config.app.data_dir / "qdrant")
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

            if reminders_cfg.enabled:
                assert parser is not None
                app.scheduler = SchedulerService(
                    SchedulerStore(app.db),
                    app.bus,
                    tz,
                    tick_seconds=config.modules.scheduler.tick_seconds,
                    misfire_grace_minutes=config.modules.scheduler.misfire_grace_minutes,
                )
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

            memory: MemoryService | None = None
            if config.modules.memory.enabled:
                # векторный recall памяти включается вместе с RAG (общий Qdrant
                # и эмбеддер); без него память работает на FTS, как в Sprint 3.
                # semantic=false снимает эмбеддинг-модель с горячего пути ради
                # экономии RAM (см. modules.memory.semantic в config)
                mem_vectors = app.vectors if config.modules.memory.semantic else None
                mem_embedder = app.gateway if config.modules.memory.semantic else None
                memory = MemoryService(
                    MemoryStore(app.db, vectors=mem_vectors, embedder=mem_embedder)
                )
                if not config.modules.memory.semantic:
                    log.info("memory_semantic_disabled", reason="config: FTS-only recall")
                for spec in build_memory_tools(memory):
                    registry.register(spec)

            processor = AgentOrchestrator(
                gateway=app.gateway,
                history=HistoryStore(app.db),
                registry=registry,
                audit=audit,
                config=config.agent,
                timezone=config.app.timezone,
                memory=memory,
                extra_commands=app._build_extra_commands(rag_service, tasks),
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

        return app

    def _notify_targets(self) -> list[tuple[str, str]]:
        """Каналы доставки уведомлений (напоминания, сводка): все активные."""
        targets: list[tuple[str, str]] = []
        telegram_cfg = self.config.channels.telegram
        if telegram_cfg.enabled and telegram_cfg.allowed_user_ids:
            targets.append(("telegram", str(telegram_cfg.allowed_user_ids[0])))
        if self.config.channels.cli.enabled:
            targets.append(("cli", "local"))
        return targets

    def _build_extra_commands(
        self, rag_service: RAGService | None, tasks: TasksService | None
    ) -> dict[str, tuple[str, Callable[[], Awaitable[str]]]]:
        """Сервис-команды модулей для оркестратора (инъекция: ядро не знает модулей)."""
        commands: dict[str, tuple[str, Callable[[], Awaitable[str]]]] = {}
        if tasks is not None:
            commands["/tasks"] = ("открытые задачи по срокам", tasks.overview_text)
        if self.reminders is not None:
            commands["/reminders"] = ("предстоящие напоминания", self.reminders.overview_text)
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
        if self.gateway is not None and self.config.llm.keep_warm_minutes > 0:
            # прогрев ДО приёма сообщений и ДО старта индексатора: первый вопрос
            # после запуска не ждёт холодную загрузку модели и не ловит ReadTimeout
            # (загрузка на CPU — минуты; в это время бот сознательно молчит)
            log.info("model_warming_up")
            # заметная строка для владельца: среди технических строк лога легко
            # пропустить момент готовности, а до него бот молчит
            print("⏳ Загружаю модель в память, подождите (обычно меньше минуты)…",
                  flush=True)
            await self.gateway.warmup()
            print("✅ Модель загружена, можно писать.", flush=True)
            log.info("model_ready")
            background.append(
                asyncio.create_task(
                    keep_warm_loop(self.gateway, self.config.llm.keep_warm_minutes * 60)
                )
            )
        if self.indexer is not None:
            background.append(asyncio.create_task(self.indexer.run_forever()))
        if self.scheduler is not None:
            # порядок важен: сначала регистрация сводки и синхронизация задач,
            # потом цикл — иначе перерегистрация может затереть созревший джоб
            if self.brief is not None:
                await self.brief.schedule()
            if self.reminders is not None:
                synced = await self.reminders.sync_open_tasks()
                if synced:
                    log.info("task_reminders_synced", tasks=synced)
            background.append(asyncio.create_task(self.scheduler.run_forever()))
        try:
            await asyncio.gather(*(ch.start() for ch in self.channels))
        finally:
            for task in background:
                task.cancel()
            await self.shutdown()

    async def shutdown(self) -> None:
        for channel in self.channels:
            await channel.stop()
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

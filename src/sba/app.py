"""Композиция приложения: конфиг → инфраструктура → ядро → каналы.

Единственное место, которому позволено знать обо всех частях системы
(DI вручную, без фреймворков — docs/02-architecture.md ADR-1).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

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
from sba.core.types import ChannelAdapter, MessageProcessor
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

log = structlog.get_logger(__name__)


async def keep_warm_loop(gateway: ModelGateway, interval_seconds: float) -> None:
    """Фоновый прогрев: не даёт Ollama выгрузить chat-модель из RAM."""
    while True:
        await gateway.warmup()
        await asyncio.sleep(interval_seconds)


class App:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db: Database | None = None
        self.bus = EventBus()
        self.router: Router | None = None
        self.gateway: ModelGateway | None = None
        self.vectors: VectorStore | None = None
        self.indexer: IndexerService | None = None
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

            memory: MemoryService | None = None
            if config.modules.memory.enabled:
                # векторный recall памяти включается вместе с RAG (общий Qdrant
                # и эмбеддер); без него память работает на FTS, как в Sprint 3
                memory = MemoryService(
                    MemoryStore(app.db, vectors=app.vectors, embedder=app.gateway)
                )
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
                extra_commands=app._build_extra_commands(rag_service),
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
            )
            app.router.register_channel(telegram)
            app.channels.append(telegram)

        return app

    def _build_extra_commands(
        self, rag_service: RAGService | None
    ) -> dict[str, tuple[str, Callable[[], Awaitable[str]]]]:
        """Сервис-команды модулей для оркестратора (инъекция: ядро не знает модулей)."""
        commands: dict[str, tuple[str, Callable[[], Awaitable[str]]]] = {}
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
            background.append(
                asyncio.create_task(
                    keep_warm_loop(self.gateway, self.config.llm.keep_warm_minutes * 60)
                )
            )
        if self.indexer is not None:
            background.append(asyncio.create_task(self.indexer.run_forever()))
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

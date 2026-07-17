"""Композиция приложения: конфиг → инфраструктура → ядро → каналы.

Единственное место, которому позволено знать обо всех частях системы
(DI вручную, без фреймворков — docs/02-architecture.md ADR-1).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import structlog

from sba import __version__
from sba.channels.cli.repl import CliChannel
from sba.channels.telegram.gateway import TelegramChannel
from sba.core.agent.orchestrator import AgentOrchestrator
from sba.core.events import EventBus
from sba.core.history import HistoryStore
from sba.core.processor import EchoProcessor
from sba.core.router import Router
from sba.core.types import ChannelAdapter, MessageProcessor
from sba.infra.config import Config, ConfigError, load_config
from sba.infra.db import Database
from sba.infra.logging import setup_logging
from sba.llm.config import load_models_config
from sba.llm.service import ModelGateway

log = structlog.get_logger(__name__)


class App:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db: Database | None = None
        self.bus = EventBus()
        self.router: Router | None = None
        self.gateway: ModelGateway | None = None
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
            app.gateway = ModelGateway(load_models_config(config_dir / "models.yaml"))
            processor = AgentOrchestrator(
                gateway=app.gateway,
                history=HistoryStore(app.db),
                config=config.agent,
                timezone=config.app.timezone,
            )

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

    async def run(self) -> None:
        """Работает, пока живы каналы (при одном интерактивном канале —
        выход из REPL завершает приложение; сервис-режим живёт на Telegram)."""
        if not self.channels:
            log.error("no_channels_enabled")
            return
        try:
            await asyncio.gather(*(ch.start() for ch in self.channels))
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        for channel in self.channels:
            await channel.stop()
        if self.gateway is not None:
            await self.gateway.aclose()
        if self.db is not None:
            await self.db.close()
        log.info("app_stopped")


async def run_app(config_dir: Path) -> None:
    app = await App.create(config_dir)
    await app.run()

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
from sba.core.events import EventBus
from sba.core.processor import EchoProcessor
from sba.core.router import Router
from sba.core.types import ChannelAdapter
from sba.infra.config import Config, load_config
from sba.infra.db import Database
from sba.infra.logging import setup_logging

log = structlog.get_logger(__name__)


class App:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.db: Database | None = None
        self.bus = EventBus()
        self.router: Router | None = None
        self.channels: list[ChannelAdapter] = []

    @classmethod
    async def create(cls, config_dir: Path) -> App:
        config = load_config(config_dir)
        setup_logging(config.logging.level, config.logging.format)
        log.info("app_starting", version=__version__, data_dir=str(config.app.data_dir))

        app = cls(config)
        app.db = await Database.open(config.app.data_dir / "sba.db")
        app.router = Router(
            db=app.db,
            bus=app.bus,
            processor=EchoProcessor(),  # Sprint 1: Agent Orchestrator
            session_idle_timeout=timedelta(minutes=config.session.idle_timeout_minutes),
        )
        if config.channels.cli.enabled:
            cli = CliChannel(handle_incoming=app.router.handle_incoming)
            app.router.register_channel(cli)
            app.channels.append(cli)
        return app

    async def run(self) -> None:
        """Работает, пока живы каналы (выход из REPL завершает приложение)."""
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
        if self.db is not None:
            await self.db.close()
        log.info("app_stopped")


async def run_app(config_dir: Path) -> None:
    app = await App.create(config_dir)
    await app.run()

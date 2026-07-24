"""App.run(): выход одного канала останавливает остальные (Ctrl+C-баг 2026-07-24).

Раньше `asyncio.gather` ждал ВСЕ каналы: Ctrl+C гасил только консоль, а Telegram
опрашивал дальше — бот не завершался. Теперь ждём первый завершившийся канал."""

from __future__ import annotations

import asyncio

from sba.app import App
from sba.infra.config import Config


class QuickChannel:
    """Как консоль на /quit или Ctrl+C: start() завершается сам."""

    name = "cli"

    def __init__(self) -> None:
        self.stopped = False

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        self.stopped = True


class ForeverChannel:
    """Как Telegram long polling: сам никогда не завершается, только по отмене."""

    name = "telegram"

    def __init__(self) -> None:
        self.stopped = False

    async def start(self) -> None:
        await asyncio.Event().wait()

    async def stop(self) -> None:
        self.stopped = True


async def test_run_stops_all_channels_when_one_exits() -> None:
    app = App(Config())
    quick, forever = QuickChannel(), ForeverChannel()
    app.channels = [forever, quick]  # forever первым — но выходит quick

    # не должно зависнуть: как только quick завершился — останавливаем forever
    await asyncio.wait_for(app.run(), timeout=3)

    assert quick.stopped
    assert forever.stopped  # бесконечный канал тоже остановлен, а не брошен

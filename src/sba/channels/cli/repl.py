"""Консольный канал (REPL) — инструмент разработки и smoke-проверок.

Тонкий адаптер: читает строки из stdin, отправляет в Router, печатает ответы.
Никакой логики обработки — только перевод «терминал ⇄ сообщения ядра».
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from sba.core.types import IncomingMessage, OutgoingKind, OutgoingMessage

PROMPT = "вы> "
EXIT_COMMANDS = {"/quit", "/exit"}


class CliChannel:
    name = "cli"

    def __init__(
        self,
        handle_incoming: Callable[[IncomingMessage], Awaitable[None]],
        user_id: str = "local",
    ) -> None:
        self._handle_incoming = handle_incoming
        self._user_id = user_id
        self._stopped = asyncio.Event()

    async def start(self) -> None:
        print("Second Brain Agent — консольный канал. /quit для выхода.")
        while not self._stopped.is_set():
            line = await asyncio.to_thread(self._read_line)
            if line is None or line.strip() in EXIT_COMMANDS:
                break
            text = line.strip()
            if not text:
                continue
            await self._handle_incoming(
                IncomingMessage(user_id=self._user_id, channel=self.name, text=text)
            )

    async def stop(self) -> None:
        self._stopped.set()

    async def send(self, out: OutgoingMessage) -> None:
        prefix = "🔔 " if out.kind == OutgoingKind.NOTIFICATION else ""
        print(f"{prefix}ассистент> {out.text}")

    def _read_line(self) -> str | None:
        try:
            return input(PROMPT)
        except EOFError:
            return None
        except KeyboardInterrupt:
            return None

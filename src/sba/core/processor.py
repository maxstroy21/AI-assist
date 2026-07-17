"""Обработчики сообщений.

EchoProcessor — заглушка Sprint 0, проверяющая сквозной путь
канал → Router → обработчик → доставка. В Sprint 1 его место занимает
Agent Orchestrator (core/agent/), контракт MessageProcessor не меняется.
"""

from __future__ import annotations

from sba.core.types import IncomingMessage, Session


class EchoProcessor:
    async def process(self, msg: IncomingMessage, session: Session) -> str:
        return f"[echo] {msg.text}"

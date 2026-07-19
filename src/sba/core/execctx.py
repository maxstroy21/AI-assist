"""Контекст исполнения текущего хода агента.

Инструменты получают только свои аргументы (ToolSpec.handler), но некоторым
нужна привязка к источнику — например, задача хранит id исходного сообщения
(DoD Sprint 5). ContextVar передаёт это сквозь agent loop без изменения
сигнатур: оркестратор ставит значение в начале обработки, инструмент читает.
"""

from __future__ import annotations

from contextvars import ContextVar

current_message_id: ContextVar[str | None] = ContextVar("current_message_id", default=None)

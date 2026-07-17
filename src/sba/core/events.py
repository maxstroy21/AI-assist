"""Внутренняя событийная шина (in-process pub/sub).

Модули реагируют на факты, не вызывая друг друга напрямую
(docs/02-architecture.md §2.2). Ошибка одного подписчика логируется
и не мешает остальным.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TypeVar

import structlog
from pydantic import BaseModel

from sba.core.types import IncomingMessage, Session

log = structlog.get_logger(__name__)


class Event(BaseModel):
    pass


class MessageReceived(Event):
    message: IncomingMessage
    session: Session


class MessageProcessed(Event):
    message: IncomingMessage
    session: Session
    reply_text: str


class SessionClosed(Event):
    session: Session


E = TypeVar("E", bound=Event)
Handler = Callable[[E], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler[Event]]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: Handler[E]) -> None:
        self._handlers[event_type].append(handler)  # type: ignore[arg-type]

    async def publish(self, event: Event) -> None:
        """Доставить событие всем подписчикам, изолируя их ошибки."""
        handlers = self._handlers.get(type(event), [])
        if not handlers:
            return
        results = await asyncio.gather(
            *(h(event) for h in handlers), return_exceptions=True
        )
        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                log.error(
                    "event_handler_failed",
                    event_type=type(event).__name__,
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    error=str(result),
                )

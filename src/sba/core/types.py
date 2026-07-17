"""Общие типы ядра: сообщения, сессии, контракты каналов и обработчика."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class MessageKind(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    DOCUMENT = "document"


class OutgoingKind(StrEnum):
    REPLY = "reply"          # ответ в канал-источник
    NOTIFICATION = "notification"  # инициатива ассистента (напоминание и т.п.)


class IncomingMessage(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: str
    channel: str
    text: str
    kind: MessageKind = MessageKind.TEXT
    created_at: datetime = Field(default_factory=utcnow)


class OutgoingMessage(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: str
    channel: str
    text: str
    kind: OutgoingKind = OutgoingKind.REPLY
    reply_to: str | None = None  # id входящего сообщения, если это ответ
    created_at: datetime = Field(default_factory=utcnow)


class Session(BaseModel):
    """Активный разговор пользователя в канале (строка conversations)."""

    id: str
    user_id: str
    channel: str
    started_at: datetime
    last_activity_at: datetime


@runtime_checkable
class ChannelAdapter(Protocol):
    """Канал общения. Знает только Router; ядро знает только этот контракт."""

    name: str

    async def start(self) -> None:
        """Долгоживущий цикл канала (long polling, REPL и т.п.)."""
        ...

    async def stop(self) -> None: ...

    async def send(self, out: OutgoingMessage) -> None: ...


class MessageProcessor(Protocol):
    """Мозг, к которому Router подключает каналы.

    Sprint 0 — эхо; со Sprint 1 здесь Agent Orchestrator.
    """

    async def process(self, msg: IncomingMessage, session: Session) -> str: ...

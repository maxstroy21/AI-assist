"""Message Router: нормализованный вход из любого канала → обработчик → доставка.

Управляет сессиями: одна активная сессия на (user_id, channel); простой
неактивности дольше таймаута закрывает сессию (событие SessionClosed)
и открывает новую.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import structlog

from sba.core.events import EventBus, MessageProcessed, MessageReceived, SessionClosed
from sba.core.types import (
    ChannelAdapter,
    IncomingMessage,
    MessageProcessor,
    OutgoingKind,
    OutgoingMessage,
    Session,
    StreamingChannel,
    new_id,
    utcnow,
)
from sba.infra.db import Database
from sba.infra.logging import bind_request_id, clear_request_context

log = structlog.get_logger(__name__)

NEW_SESSION_COMMANDS = {"/new", "/reset"}


class Router:
    def __init__(
        self,
        db: Database,
        bus: EventBus,
        processor: MessageProcessor,
        session_idle_timeout: timedelta,
    ) -> None:
        self._db = db
        self._bus = bus
        self._processor = processor
        self._idle_timeout = session_idle_timeout
        self._channels: dict[str, ChannelAdapter] = {}

    def register_channel(self, channel: ChannelAdapter) -> None:
        self._channels[channel.name] = channel
        log.info("channel_registered", channel=channel.name)

    async def handle_incoming(self, msg: IncomingMessage) -> None:
        bind_request_id(msg.id)
        try:
            if msg.text.strip().lower() in NEW_SESSION_COMMANDS:
                await self._start_fresh_session(msg)
                return
            session = await self._get_or_create_session(msg)
            await self._store_message(session, msg.id, "user", msg.kind.value, msg.text)
            await self._bus.publish(MessageReceived(message=msg, session=session))

            reply = await self._processor.process(msg, session)

            out = OutgoingMessage(
                user_id=msg.user_id,
                channel=msg.channel,
                text=reply if isinstance(reply, str) else "",
                kind=OutgoingKind.REPLY,
                reply_to=msg.id,
            )
            if isinstance(reply, str):
                await self.deliver(out)
            else:
                out.text = await self._deliver_stream(out, reply)
            await self._store_message(session, out.id, "assistant", "text", out.text)
            await self._bus.publish(
                MessageProcessed(message=msg, session=session, reply_text=out.text)
            )
        finally:
            clear_request_context()

    async def _deliver_stream(self, out: OutgoingMessage, deltas: AsyncIterator[str]) -> str:
        """Потоковая доставка; канал без send_stream получает собранный текст целиком."""
        channel = self._channels.get(out.channel)
        if channel is not None and isinstance(channel, StreamingChannel):
            return await channel.send_stream(out, deltas)
        text = "".join([delta async for delta in deltas])
        out.text = text
        if channel is None:
            log.error("channel_not_registered", channel=out.channel, message_id=out.id)
        else:
            await channel.send(out)
        return text

    async def deliver(self, out: OutgoingMessage) -> None:
        channel = self._channels.get(out.channel)
        if channel is None:
            log.error("channel_not_registered", channel=out.channel, message_id=out.id)
            return
        await channel.send(out)

    # ── сессии ────────────────────────────────────────────────────────────────

    async def _start_fresh_session(self, msg: IncomingMessage) -> None:
        """/new: закрыть активный разговор — история больше не попадает в контекст."""
        row = await self._db.fetch_one(
            "SELECT * FROM conversations"
            " WHERE user_id=? AND channel=? AND closed_at IS NULL"
            " ORDER BY started_at DESC LIMIT 1",
            (msg.user_id, msg.channel),
        )
        if row is not None:
            session = Session(
                id=row["id"],
                user_id=row["user_id"],
                channel=row["channel"],
                started_at=datetime.fromisoformat(row["started_at"]),
                last_activity_at=datetime.fromisoformat(row["last_activity_at"]),
            )
            await self._close_session(session, utcnow())
        await self.deliver(
            OutgoingMessage(
                user_id=msg.user_id,
                channel=msg.channel,
                text="🆕 Начат новый разговор — прошлая история отложена и на ответы "
                "больше не влияет.",
            )
        )

    async def _get_or_create_session(self, msg: IncomingMessage) -> Session:
        row = await self._db.fetch_one(
            "SELECT * FROM conversations"
            " WHERE user_id=? AND channel=? AND closed_at IS NULL"
            " ORDER BY started_at DESC LIMIT 1",
            (msg.user_id, msg.channel),
        )
        now = utcnow()
        if row is not None:
            session = Session(
                id=row["id"],
                user_id=row["user_id"],
                channel=row["channel"],
                started_at=datetime.fromisoformat(row["started_at"]),
                last_activity_at=datetime.fromisoformat(row["last_activity_at"]),
            )
            if now - session.last_activity_at <= self._idle_timeout:
                await self._db.execute(
                    "UPDATE conversations SET last_activity_at=? WHERE id=?",
                    (now.isoformat(), session.id),
                )
                return session.model_copy(update={"last_activity_at": now})
            await self._close_session(session, now)

        session = Session(
            id=new_id(),
            user_id=msg.user_id,
            channel=msg.channel,
            started_at=now,
            last_activity_at=now,
        )
        await self._db.execute(
            "INSERT INTO conversations (id, user_id, channel, started_at, last_activity_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (session.id, session.user_id, session.channel, now.isoformat(), now.isoformat()),
        )
        log.info("session_started", session_id=session.id, user_id=session.user_id)
        return session

    async def _close_session(self, session: Session, closed_at: datetime) -> None:
        await self._db.execute(
            "UPDATE conversations SET closed_at=? WHERE id=?",
            (closed_at.isoformat(), session.id),
        )
        log.info("session_closed", session_id=session.id)
        await self._bus.publish(SessionClosed(session=session))

    async def _store_message(
        self, session: Session, msg_id: str, role: str, kind: str, content: str
    ) -> None:
        await self._db.execute(
            "INSERT INTO messages (id, conversation_id, role, kind, content, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, session.id, role, kind, content, utcnow().isoformat()),
        )

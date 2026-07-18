from datetime import timedelta
from pathlib import Path

import pytest

from sba.core.events import EventBus, SessionClosed
from sba.core.processor import EchoProcessor
from sba.core.router import Router
from sba.core.types import IncomingMessage, OutgoingMessage
from sba.infra.db import Database


class CollectingChannel:
    name = "cli"

    def __init__(self) -> None:
        self.sent: list[OutgoingMessage] = []

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def send(self, out: OutgoingMessage) -> None:
        self.sent.append(out)


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_router(db: Database, bus: EventBus | None = None, timeout_minutes: int = 60) -> Router:
    return Router(
        db=db,
        bus=bus or EventBus(),
        processor=EchoProcessor(),
        session_idle_timeout=timedelta(minutes=timeout_minutes),
    )


async def test_echo_roundtrip(db: Database) -> None:
    router = make_router(db)
    channel = CollectingChannel()
    router.register_channel(channel)

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="привет"))

    assert len(channel.sent) == 1
    assert channel.sent[0].text == "[echo] привет"
    assert channel.sent[0].reply_to is not None


async def test_messages_persisted(db: Database) -> None:
    router = make_router(db)
    router.register_channel(CollectingChannel())

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="раз"))
    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="два"))

    rows = await db.fetch_all("SELECT role, content FROM messages ORDER BY created_at")
    assert [(r["role"], r["content"]) for r in rows] == [
        ("user", "раз"),
        ("assistant", "[echo] раз"),
        ("user", "два"),
        ("assistant", "[echo] два"),
    ]


async def test_session_reused_within_timeout(db: Database) -> None:
    router = make_router(db)
    router.register_channel(CollectingChannel())

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="раз"))
    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="два"))

    rows = await db.fetch_all("SELECT id FROM conversations")
    assert len(rows) == 1


async def test_idle_timeout_closes_session_and_publishes_event(db: Database) -> None:
    bus = EventBus()
    closed: list[SessionClosed] = []

    async def on_closed(event: SessionClosed) -> None:
        closed.append(event)

    bus.subscribe(SessionClosed, on_closed)
    router = make_router(db, bus=bus, timeout_minutes=30)
    router.register_channel(CollectingChannel())

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="раз"))
    # состарить сессию «в прошлое» глубже таймаута
    await db.execute(
        "UPDATE conversations SET last_activity_at=? WHERE closed_at IS NULL",
        ("2020-01-01T00:00:00+00:00",),
    )
    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="два"))

    conversations = await db.fetch_all("SELECT closed_at FROM conversations")
    assert len(conversations) == 2
    assert len(closed) == 1
    assert sum(1 for r in conversations if r["closed_at"] is None) == 1


async def test_new_command_closes_session(db: Database) -> None:
    router = make_router(db)
    channel = CollectingChannel()
    router.register_channel(channel)

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="раз"))
    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="/new"))
    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="два"))

    conversations = await db.fetch_all("SELECT closed_at FROM conversations")
    assert len(conversations) == 2
    assert sum(1 for r in conversations if r["closed_at"] is None) == 1
    # подтверждение /new доставлено, но в историю не записано
    assert any("новый разговор" in m.text.lower() for m in channel.sent)
    rows = await db.fetch_all("SELECT content FROM messages")
    assert all("/new" != r["content"] for r in rows)


async def test_unknown_channel_does_not_crash(db: Database) -> None:
    router = make_router(db)  # канал не зарегистрирован
    await router.handle_incoming(IncomingMessage(user_id="u1", channel="ghost", text="эй"))

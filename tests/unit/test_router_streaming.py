from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

from sba.core.events import EventBus, MessageProcessed
from sba.core.router import Router
from sba.core.types import IncomingMessage, OutgoingMessage, Reply, Session
from sba.infra.db import Database


class StreamingProcessor:
    async def process(self, msg: IncomingMessage, session: Session) -> Reply:
        async def deltas() -> AsyncIterator[str]:
            for part in ("по", "ток", "!"):
                yield part

        return deltas()


class PlainChannel:
    name = "cli"

    def __init__(self) -> None:
        self.sent: list[OutgoingMessage] = []

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def send(self, out: OutgoingMessage) -> None:
        self.sent.append(out)


class StreamCapableChannel(PlainChannel):
    def __init__(self) -> None:
        super().__init__()
        self.streamed: list[str] = []

    async def send_stream(self, out: OutgoingMessage, deltas: AsyncIterator[str]) -> str:
        parts = [d async for d in deltas]
        self.streamed.extend(parts)
        return "".join(parts)


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_router(db: Database, bus: EventBus) -> Router:
    return Router(
        db=db,
        bus=bus,
        processor=StreamingProcessor(),
        session_idle_timeout=timedelta(minutes=60),
    )


async def test_streaming_channel_receives_deltas_and_full_text_persisted(db: Database) -> None:
    bus = EventBus()
    processed: list[MessageProcessed] = []

    async def on_processed(event: MessageProcessed) -> None:
        processed.append(event)

    bus.subscribe(MessageProcessed, on_processed)
    router = make_router(db, bus)
    channel = StreamCapableChannel()
    router.register_channel(channel)

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="ну"))

    assert channel.streamed == ["по", "ток", "!"]
    rows = await db.fetch_all("SELECT content FROM messages WHERE role='assistant'")
    assert [r["content"] for r in rows] == ["поток!"]
    assert processed[0].reply_text == "поток!"


async def test_plain_channel_gets_accumulated_text(db: Database) -> None:
    router = make_router(db, EventBus())
    channel = PlainChannel()
    router.register_channel(channel)

    await router.handle_incoming(IncomingMessage(user_id="u1", channel="cli", text="ну"))

    assert len(channel.sent) == 1
    assert channel.sent[0].text == "поток!"

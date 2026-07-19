"""Маршрутизация кнопок в Router и клавиатура Telegram (Sprint 6)."""

from datetime import timedelta
from pathlib import Path

import pytest

from sba.channels.telegram.gateway import TelegramChannel
from sba.core.events import EventBus
from sba.core.router import Router
from sba.core.types import IncomingMessage, MessageAction, OutgoingMessage, Session
from sba.infra.db import Database


class EchoProcessorStub:
    async def process(self, msg: IncomingMessage, session: Session) -> str:
        return msg.text


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


@pytest.fixture
def router(db: Database) -> Router:
    return Router(
        db=db,
        bus=EventBus(),
        processor=EchoProcessorStub(),
        session_idle_timeout=timedelta(minutes=60),
    )


async def test_action_dispatched_by_prefix(router: Router) -> None:
    seen: list[tuple[str, str]] = []

    async def handler(user_id: str, action_id: str) -> str:
        seen.append((user_id, action_id))
        return "готово"

    router.register_action_handler("rem", handler)
    ack = await router.handle_action("42", "telegram", "rem:done:abc")
    assert ack == "готово"
    assert seen == [("42", "rem:done:abc")]


async def test_unknown_prefix_gives_polite_answer(router: Router) -> None:
    ack = await router.handle_action("42", "telegram", "ghost:x")
    assert "устарела" in ack


async def test_handler_error_does_not_crash_channel(router: Router) -> None:
    async def broken(user_id: str, action_id: str) -> str:
        raise RuntimeError("boom")

    router.register_action_handler("rem", broken)
    ack = await router.handle_action("42", "telegram", "rem:done:abc")
    assert ack.startswith("⚠️")


def test_telegram_keyboard_from_actions() -> None:
    out = OutgoingMessage(
        user_id="1",
        channel="telegram",
        text="напоминание",
        actions=[
            MessageAction(id="rem:done:a", label="✅ Сделал"),
            MessageAction(id="rem:snooze:a", label="⏰ Позже"),
        ],
    )
    markup = TelegramChannel._keyboard(out)
    assert markup is not None
    row = markup.inline_keyboard[0]
    assert [b.text for b in row] == ["✅ Сделал", "⏰ Позже"]
    assert [b.callback_data for b in row] == ["rem:done:a", "rem:snooze:a"]

    plain = OutgoingMessage(user_id="1", channel="telegram", text="без кнопок")
    assert TelegramChannel._keyboard(plain) is None

"""Web-чат (Sprint 9): протокол WS, streaming, кнопки, история.

Канал тестируется напрямую с фейковым WebSocket (duck typing): логика
протокола не зависит от транспорта uvicorn.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import WebSocketDisconnect

from sba.channels.webchat.gateway import WebChatChannel
from sba.core.types import (
    IncomingMessage,
    MessageAction,
    OutgoingKind,
    OutgoingMessage,
)


class FakeWS:
    """Минимальный дубль WebSocket: очередь входящих, список отправленного."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        item = await self._incoming.get()
        if item is None:
            raise WebSocketDisconnect(1000)
        return item

    def push(self, payload: dict) -> None:
        self._incoming.put_nowait(json.dumps(payload))

    def push_raw(self, raw: str) -> None:
        self._incoming.put_nowait(raw)

    def close(self) -> None:
        self._incoming.put_nowait(None)


def make_channel(**overrides):
    incoming: list[IncomingMessage] = []
    actions: list[str] = []

    async def handle_incoming(msg: IncomingMessage) -> None:
        incoming.append(msg)

    async def handle_action(user_id: str, channel: str, action_id: str) -> str:
        actions.append(action_id)
        return f"готово: {action_id}"

    async def fetch_history() -> list[dict[str, str]]:
        return [{"role": "user", "text": "привет"}, {"role": "assistant", "text": "здравствуйте"}]

    params = dict(
        host="127.0.0.1",
        port=0,
        handle_incoming=handle_incoming,
        handle_action=handle_action,
        fetch_history=fetch_history,
    )
    params.update(overrides)
    channel = WebChatChannel(**params)
    return channel, incoming, actions


async def connected(channel, ws: FakeWS) -> asyncio.Task:
    task = asyncio.ensure_future(channel._serve_client(ws))
    for _ in range(100):
        if ws.sent:  # история пришла — соединение готово
            return task
        await asyncio.sleep(0.01)
    raise AssertionError("клиент не получил историю")


async def test_history_on_connect():
    channel, _, _ = make_channel()
    ws = FakeWS()
    task = await connected(channel, ws)
    assert ws.sent[0]["type"] == "history"
    assert [m["text"] for m in ws.sent[0]["messages"]] == ["привет", "здравствуйте"]
    ws.close()
    await task


async def test_message_goes_to_router():
    channel, incoming, _ = make_channel()
    ws = FakeWS()
    task = await connected(channel, ws)
    ws.push({"type": "message", "text": "сколько времени?"})
    for _ in range(100):
        if incoming:
            break
        await asyncio.sleep(0.01)
    assert incoming[0].text == "сколько времени?"
    assert incoming[0].channel == "web"
    assert incoming[0].user_id == "local"
    ws.close()
    await task


async def test_empty_and_broken_payloads_ignored():
    channel, incoming, _ = make_channel()
    ws = FakeWS()
    task = await connected(channel, ws)
    ws.push({"type": "message", "text": "   "})
    ws.push_raw("не json вовсе")
    ws.push({"type": "неизвестный"})
    await asyncio.sleep(0.05)
    assert not incoming
    ws.close()
    await task


async def test_send_broadcasts_with_actions():
    channel, _, _ = make_channel()
    ws1, ws2 = FakeWS(), FakeWS()
    t1 = await connected(channel, ws1)
    t2 = await connected(channel, ws2)
    await channel.send(
        OutgoingMessage(
            user_id="local",
            channel="web",
            text="Напоминание: позвонить",
            kind=OutgoingKind.NOTIFICATION,
            actions=[MessageAction(id="rem:done:1", label="✅ Сделал")],
        )
    )
    for ws in (ws1, ws2):
        msg = ws.sent[-1]
        assert msg["type"] == "message"
        assert msg["kind"] == "notification"
        assert msg["actions"] == [{"id": "rem:done:1", "label": "✅ Сделал"}]
    ws1.close(), ws2.close()
    await t1
    await t2


async def test_send_stream_order_and_full_text():
    channel, _, _ = make_channel()
    ws = FakeWS()
    task = await connected(channel, ws)

    async def deltas():
        yield "Здрав"
        yield "ствуйте"

    out = OutgoingMessage(user_id="local", channel="web", text="")
    text = await channel.send_stream(out, deltas())
    assert text == "Здравствуйте"
    kinds = [m["type"] for m in ws.sent[1:]]
    assert kinds == ["stream_start", "delta", "delta", "stream_end"]
    assert ws.sent[-1]["text"] == "Здравствуйте"
    assert all(m.get("msg_id") == out.id for m in ws.sent[1:])
    ws.close()
    await task


async def test_action_ack_broadcast():
    channel, _, actions = make_channel()
    ws = FakeWS()
    task = await connected(channel, ws)
    ws.push({"type": "action", "id": "rem:done:42"})
    for _ in range(100):
        if len(ws.sent) > 1:
            break
        await asyncio.sleep(0.01)
    assert actions == ["rem:done:42"]
    ack = ws.sent[-1]
    assert ack["type"] == "action_result"
    assert ack["action_id"] == "rem:done:42"
    assert "готово" in ack["text"]
    ws.close()
    await task


async def test_dead_client_dropped_on_broadcast():
    channel, _, _ = make_channel()

    class BrokenWS(FakeWS):
        async def send_text(self, data: str) -> None:
            raise RuntimeError("соединение закрыто")

    broken = BrokenWS()
    channel._clients.add(broken)
    await channel._broadcast({"type": "message", "text": "x"})
    assert broken not in channel._clients


@pytest.mark.parametrize("payload", [{"type": "message"}, {"type": "action"}])
async def test_missing_fields_ignored(payload):
    channel, incoming, actions = make_channel()
    ws = FakeWS()
    task = await connected(channel, ws)
    ws.push(payload)
    await asyncio.sleep(0.05)
    assert not incoming and not actions
    ws.close()
    await task

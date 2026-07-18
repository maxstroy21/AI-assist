import asyncio
from collections.abc import AsyncIterator

from sba.channels.telegram.gateway import with_heartbeat


async def slow_source() -> AsyncIterator[str]:
    yield "быстрый"
    await asyncio.sleep(0.08)
    yield "медленный"


async def test_heartbeat_emits_wait_during_silence() -> None:
    events = [e async for e in with_heartbeat(slow_source(), interval=0.03)]
    kinds = [k for k, _ in events]
    assert kinds[0] == "text"
    assert "wait" in kinds  # во время паузы были сигналы ожидания
    assert kinds[-1] == "text"
    texts = [p for k, p in events if k == "text"]
    assert texts == ["быстрый", "медленный"]


async def test_heartbeat_waits_accumulate() -> None:
    async def silent_then_text() -> AsyncIterator[str]:
        await asyncio.sleep(0.1)
        yield "готово"

    events = [e async for e in with_heartbeat(silent_then_text(), interval=0.03)]
    waits = [int(p) for k, p in events if k == "wait"]
    assert waits == sorted(waits)  # счётчик секунд нарастает
    assert events[-1] == ("text", "готово")

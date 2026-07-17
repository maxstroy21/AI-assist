from sba.core.events import EventBus, MessageProcessed, MessageReceived
from sba.core.types import IncomingMessage, Session, utcnow


def make_session() -> Session:
    now = utcnow()
    return Session(id="s1", user_id="u1", channel="cli", started_at=now, last_activity_at=now)


def make_event() -> MessageReceived:
    msg = IncomingMessage(user_id="u1", channel="cli", text="hi")
    return MessageReceived(message=msg, session=make_session())


async def test_publish_reaches_all_subscribers() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def h1(event: MessageReceived) -> None:
        seen.append("h1")

    async def h2(event: MessageReceived) -> None:
        seen.append("h2")

    bus.subscribe(MessageReceived, h1)
    bus.subscribe(MessageReceived, h2)
    await bus.publish(make_event())
    assert sorted(seen) == ["h1", "h2"]


async def test_failing_handler_does_not_break_others() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def bad(event: MessageReceived) -> None:
        raise RuntimeError("boom")

    async def good(event: MessageReceived) -> None:
        seen.append("good")

    bus.subscribe(MessageReceived, bad)
    bus.subscribe(MessageReceived, good)
    await bus.publish(make_event())  # не должно бросить
    assert seen == ["good"]


async def test_no_subscribers_is_fine() -> None:
    bus = EventBus()
    await bus.publish(make_event())


async def test_subscribers_filtered_by_event_type() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def only_processed(event: MessageProcessed) -> None:
        seen.append("processed")

    bus.subscribe(MessageProcessed, only_processed)
    await bus.publish(make_event())  # MessageReceived — не для этого подписчика
    assert seen == []

"""Health-мониторинг (Sprint 10): heartbeat, тревога о зависании и восстановление."""

from __future__ import annotations

from sba.core.types import OutgoingKind, OutgoingMessage
from sba.modules.health.service import HealthMonitor, _human_duration


class Clock:
    """Управляемое монотонное время для тестов (реальные паузы не нужны)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_monitor() -> tuple[HealthMonitor, list[OutgoingMessage], Clock]:
    sent: list[OutgoingMessage] = []

    async def deliver(out: OutgoingMessage) -> None:
        sent.append(out)

    clock = Clock()
    monitor = HealthMonitor(
        deliver=deliver,
        targets=[("telegram", "909506738"), ("cli", "local")],
        check_interval_seconds=60,
        clock=clock,
    )
    return monitor, sent, clock


async def test_no_alert_while_component_beats() -> None:
    monitor, sent, clock = make_monitor()
    beat = monitor.register("indexer", "индексатор", max_silence_seconds=300)
    for _ in range(5):
        clock.t += 100  # меньше порога между сигналами
        beat()
        await monitor.check_once()
    assert sent == []  # исправно бьётся — тревоги нет


async def test_alert_once_on_stall_then_recovery() -> None:
    monitor, sent, clock = make_monitor()
    beat = monitor.register("indexer", "индексатор документов", max_silence_seconds=300)

    clock.t += 400  # молчит дольше порога
    await monitor.check_once()
    await monitor.check_once()  # повторная проверка не должна слать второе уведомление
    assert len(sent) == 2  # одно уведомление × два целевых канала
    assert all(m.kind == OutgoingKind.NOTIFICATION for m in sent)
    assert "завис" in sent[0].text
    assert "индексатор документов" in sent[0].text
    assert {m.channel for m in sent} == {"telegram", "cli"}

    sent.clear()
    beat()  # компонент снова подал сигнал
    await monitor.check_once()
    assert len(sent) == 2  # уведомление о восстановлении, снова в оба канала
    assert "снова работает" in sent[0].text

    sent.clear()
    await monitor.check_once()  # восстановление уже сообщили — тишина
    assert sent == []


async def test_overview_marks_states() -> None:
    monitor, _sent, clock = make_monitor()
    beat_ok = monitor.register("scheduler", "планировщик", max_silence_seconds=300)
    monitor.register("indexer", "индексатор", max_silence_seconds=300)  # ни разу не бился

    beat_ok()
    clock.t += 10
    text = await monitor.overview_text()
    assert "планировщик: ✅ работает" in text
    assert "индексатор: ⏳ запускается" in text

    clock.t += 1000  # оба замолчали
    stalled = await monitor.overview_text()
    assert "планировщик: ⚠️ завис" in stalled


async def test_empty_monitor_overview_is_honest() -> None:
    monitor, _sent, _clock = make_monitor()
    assert "нет отслеживаемых" in await monitor.overview_text()


async def test_notify_survives_one_broken_channel() -> None:
    sent: list[str] = []

    async def deliver(out: OutgoingMessage) -> None:
        if out.channel == "telegram":
            raise RuntimeError("телега легла")
        sent.append(out.channel)

    clock = Clock()
    monitor = HealthMonitor(
        deliver=deliver,
        targets=[("telegram", "1"), ("cli", "local")],
        check_interval_seconds=60,
        clock=clock,
    )
    monitor.register("indexer", "индексатор", max_silence_seconds=100)
    clock.t += 200
    await monitor.check_once()  # не должно упасть из-за телеги
    assert sent == ["cli"]  # второй канал всё равно получил


def test_human_duration_formats() -> None:
    assert _human_duration(30) == "30 с"
    assert _human_duration(5 * 60) == "5 мин"
    assert _human_duration(2 * 3600) == "2 ч"
    assert _human_duration(2 * 3600 + 5 * 60) == "2 ч 5 мин"

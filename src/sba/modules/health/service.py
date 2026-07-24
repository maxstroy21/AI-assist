"""Health-мониторинг с самоотчётом в Telegram (Sprint 10, шаг 2).

Фоновые компоненты (индексатор, планировщик, консолидация памяти) на каждом
витке своего цикла подают «сигнал жизни» (heartbeat). Монитор периодически
смотрит, не замолчал ли кто дольше отпущенного окна, и при «зависании» шлёт
владельцу уведомление во все активные каналы («индексатор молчит 2 часа»);
когда сигналы возобновляются — сообщает, что компонент снова работает. Одно
уведомление на эпизод — без спама. Ревизия — команда /health мимо LLM.

Почему heartbeat, а не проверка «изнутри»: молчание на витке цикла одинаково
ловит и зависший `await`, и внезапно умершую фоновую задачу — не нужно знать,
как именно сломалось. Порог молчания задаёт композиция (app.py) из интервала
самого компонента, чтобы нормальная пауза цикла не считалась сбоем.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from sba.core.types import OutgoingKind, OutgoingMessage

log = structlog.get_logger(__name__)

Deliver = Callable[[OutgoingMessage], Awaitable[None]]


@dataclass
class _Component:
    key: str            # латиница для логов: indexer / scheduler / …
    title: str          # человеческое имя для владельца: «индексатор»
    max_silence: float  # секунд молчания = «завис»
    last_beat: float    # time.monotonic() последнего сигнала
    alerted: bool = False   # уже уведомили о зависании (чтобы не спамить)
    ever_beat: bool = False  # был ли хоть один сигнал (отличаем «запускается»)


class HealthMonitor:
    def __init__(
        self,
        deliver: Deliver,
        targets: list[tuple[str, str]],
        check_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._deliver = deliver
        self._targets = targets
        self._check_interval = check_interval_seconds
        self._clock = clock
        self._components: dict[str, _Component] = {}

    def register(self, key: str, title: str, max_silence_seconds: float) -> Callable[[], None]:
        """Регистрирует компонент и возвращает его функцию heartbeat.

        Стартовый last_beat = сейчас: компонент получает полное окно на первый
        сигнал (иначе тревога сработала бы сразу на старте)."""
        comp = _Component(
            key=key, title=title, max_silence=max_silence_seconds, last_beat=self._clock()
        )
        self._components[key] = comp

        def beat() -> None:
            comp.last_beat = self._clock()
            comp.ever_beat = True

        return beat

    async def check_once(self) -> None:
        now = self._clock()
        for comp in self._components.values():
            silent = now - comp.last_beat
            if silent > comp.max_silence and not comp.alerted:
                comp.alerted = True
                log.warning(
                    "health_component_stalled", component=comp.key, silent_seconds=int(silent)
                )
                await self._notify(
                    f"⚠️ Похоже, «{comp.title}» завис: нет признаков работы уже "
                    f"{_human_duration(silent)}. Возможно, стоит перезапустить ассистента."
                )
            elif silent <= comp.max_silence and comp.alerted:
                comp.alerted = False
                log.info("health_component_recovered", component=comp.key)
                await self._notify(f"✅ «{comp.title}» снова работает.")

    async def run_forever(self) -> None:
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                await self.check_once()
            except Exception as exc:  # монитор не должен умирать ни от чего
                log.error("health_check_failed", error=str(exc))

    async def _notify(self, text: str) -> None:
        for channel, user in self._targets:
            try:
                await self._deliver(
                    OutgoingMessage(
                        user_id=user, channel=channel, text=text,
                        kind=OutgoingKind.NOTIFICATION,
                    )
                )
            except Exception as exc:  # один недоступный канал не глушит остальные
                log.error("health_notify_failed", channel=channel, error=str(exc))

    async def overview_text(self) -> str:
        """Состояние компонентов для команды /health (мимо LLM)."""
        now = self._clock()
        lines = ["🩺 Состояние компонентов:"]
        for comp in self._components.values():
            silent = now - comp.last_beat
            if silent > comp.max_silence:
                mark = "⚠️ завис"
            elif not comp.ever_beat:
                mark = "⏳ запускается"
            else:
                mark = "✅ работает"
            lines.append(
                f"• {comp.title}: {mark} (последний сигнал {_human_duration(silent)} назад)"
            )
        if len(self._components) == 0:
            lines.append("• нет отслеживаемых фоновых компонентов")
        return "\n".join(lines)


def _human_duration(seconds: float) -> str:
    total = int(seconds)
    if total < 60:
        return f"{total} с"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes} мин"
    hours, rem = divmod(minutes, 60)
    return f"{hours} ч {rem} мин" if rem else f"{hours} ч"

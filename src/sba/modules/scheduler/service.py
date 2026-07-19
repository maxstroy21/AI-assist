"""Scheduler: отложенные и повторяющиеся события, переживающие рестарт.

Отступление от ADR-9 (см. docs/06-roadmap.md, Sprint 6): вместо APScheduler —
свой цикл поверх уже имеющегося SQLite. Причины: не тащить SQLAlchemy ради
job store, полный контроль misfire-политики, джобы не пиклятся (JSON), и
объём — сотня строк против новой большой зависимости.

Джоб только публикует событие JobFired в шину — сам ничего не исполняет
(слабая связанность, docs/04-services.md §12). Семантика — at-least-once:
после сбоя между публикацией и продвижением джоба событие повторится,
идемпотентность обеспечивает получатель (reminder_log).

Misfire (срабатывание пропущено — приложение не работало):
- deliver — доставить с опозданием (напоминания: лучше поздно, чем никогда);
- skip — в пределах grace доставить, позже пропустить и перейти к следующему
  срабатыванию (утренняя сводка: вечером она уже не нужна).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, tzinfo

import structlog

from sba.core.events import Event, EventBus
from sba.modules.scheduler.store import JobView, SchedulerStore
from sba.modules.tasks.dates import next_occurrence

log = structlog.get_logger(__name__)


class JobFired(Event):
    job_id: str
    topic: str
    payload: dict[str, str]
    scheduled_for: datetime   # aware: на какой момент было назначено срабатывание


class SchedulerService:
    def __init__(
        self,
        store: SchedulerStore,
        bus: EventBus,
        timezone: tzinfo,
        tick_seconds: float = 15.0,
        misfire_grace_minutes: float = 240.0,
    ) -> None:
        self._store = store
        self._bus = bus
        self._tz = timezone
        self._tick_seconds = tick_seconds
        self._grace = timedelta(minutes=misfire_grace_minutes)

    # ── постановка и отмена ──────────────────────────────────────────────────

    async def schedule_once(
        self,
        topic: str,
        payload: dict[str, str],
        at: datetime,
        job_id: str | None = None,
        misfire: str = "deliver",
    ) -> str:
        return await self._store.upsert(job_id, topic, payload, at, misfire=misfire)

    async def schedule_rrule(
        self,
        topic: str,
        payload: dict[str, str],
        rrule: str,
        dtstart: datetime,
        job_id: str | None = None,
        misfire: str = "deliver",
    ) -> str | None:
        """Повторяющийся джоб; None — правило не даёт срабатываний в будущем."""
        first = next_occurrence(rrule, dtstart, datetime.now(self._tz))
        if first is None:
            return None
        return await self._store.upsert(
            job_id, topic, payload, first, rrule=rrule, dtstart=dtstart, misfire=misfire
        )

    async def cancel(self, job_id: str) -> None:
        await self._store.delete(job_id)

    async def get(self, job_id: str) -> JobView | None:
        return await self._store.get(job_id)

    # ── цикл ─────────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as exc:  # цикл не должен умирать от одного сбоя
                log.error("scheduler_tick_failed", error=str(exc))
            await asyncio.sleep(self._tick_seconds)

    async def tick(self, now: datetime | None = None) -> int:
        """Обработать созревшие джобы; возвращает число опубликованных событий."""
        now = now or datetime.now(UTC)
        fired = 0
        for job in await self._store.due(now):
            late = now - job.next_fire_at
            skip = job.misfire == "skip" and late > self._grace
            if skip:
                log.info(
                    "job_misfire_skipped",
                    job_id=job.id,
                    topic=job.topic,
                    late_seconds=int(late.total_seconds()),
                )
            else:
                await self._bus.publish(
                    JobFired(
                        job_id=job.id,
                        topic=job.topic,
                        payload=job.payload,
                        scheduled_for=job.next_fire_at,
                    )
                )
                fired += 1
            await self._advance(job, now)
        return fired

    async def _advance(self, job: JobView, now: datetime) -> None:
        """Продвинуть повторяющийся джоб / убрать одноразовый — но только если
        обработчик события не перепланировал его сам (условные операции)."""
        if job.rrule is None:
            await self._store.delete_if_unchanged(job.id, job.next_fire_at)
            return
        dtstart = job.dtstart or job.next_fire_at.astimezone(self._tz)
        nxt = next_occurrence(job.rrule, dtstart, now.astimezone(self._tz))
        if nxt is None:
            await self._store.delete_if_unchanged(job.id, job.next_fire_at)
            return
        await self._store.advance_if_unchanged(job.id, job.next_fire_at, nxt)

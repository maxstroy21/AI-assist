"""Инструменты напоминаний (FR-4.2, FR-4.3, FR-5.2, FR-5.3).

Та же линия недоверия, что и у задач: id и сроки — только из ответа
инструмента; неоднозначная ссылка → кандидаты и просьба уточнить.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sba.core import execctx
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.modules.reminders.service import ReminderService
from sba.modules.reminders.store import ReminderView

WHEN_DESCRIPTION = (
    "Когда напомнить, словами пользователя: 'завтра в 15:00', 'через час', "
    "'каждый вторник в 10'. Пусто — ближайшее утро"
)


class CreateReminderArgs(BaseModel):
    text: str = Field(description="О чём напомнить, без слов 'напомни мне'")
    when: str = Field(default="", description=WHEN_DESCRIPTION)
    repeat_until_done: bool = Field(
        default=False,
        description="true, если пользователь просил напоминать снова, пока не сделает",
    )


class ListRemindersArgs(BaseModel):
    pass


class SnoozeReminderArgs(BaseModel):
    reminder: str = Field(description="Какое напоминание перенести: id вида a1b2c3 "
                          "или несколько слов из текста")
    when: str = Field(default="", description="Новый срок словами; пусто — на пару часов позже")


class CancelReminderArgs(BaseModel):
    reminder: str = Field(description="Какое напоминание отменить: id вида a1b2c3 "
                          "или несколько слов из текста")


def _ambiguous(service: ReminderService, candidates: list[ReminderView]) -> str:
    lines = [f"• {service.format_reminder(r)}" for r in candidates]
    return "Нашлось несколько подходящих напоминаний — уточни по id:\n" + "\n".join(lines)


def build_reminder_tools(service: ReminderService) -> list[ToolSpec]:
    async def create_reminder(args: BaseModel) -> str:
        assert isinstance(args, CreateReminderArgs)
        outcome = await service.create(
            text=args.text.strip(),
            when=args.when,
            repeat_until_done=args.repeat_until_done,
            source_message_id=execctx.current_message_id.get(),
        )
        if outcome.question is not None:
            return (
                "❓ Напоминание НЕ создано, нужно уточнение у пользователя: "
                f"{outcome.question}"
            )
        if outcome.reminder is None:
            return f"Напоминание НЕ создано: {outcome.error}"
        text = (
            "Создал напоминание (придёт само в срок): "
            f"{service.format_reminder(outcome.reminder)}"
        )
        if not args.when.strip():
            text += "\n(срок не был назван — выбрал ближайшее утро)"
        return text

    async def list_reminders(args: BaseModel) -> str:
        assert isinstance(args, ListRemindersArgs)
        reminders = await service.list_active()
        if not reminders:
            return (
                "Предстоящих напоминаний нет. Так и скажи пользователю — "
                "не выдумывай напоминания."
            )
        lines = [f"• {service.format_reminder(r)}" for r in reminders]
        return "Предстоящие напоминания (перечисляй только их, с id):\n" + "\n".join(lines)

    async def snooze_reminder(args: BaseModel) -> str:
        assert isinstance(args, SnoozeReminderArgs)
        candidates = await service.resolve(args.reminder)
        if not candidates:
            return (
                f"Напоминание по ссылке {args.reminder!r} не найдено. "
                "Посмотри список через list_reminders и используй id."
            )
        if len(candidates) > 1:
            return _ambiguous(service, candidates)
        reply, question = await service.snooze(candidates[0], args.when)
        if question is not None:
            return f"❓ Напоминание НЕ перенесено, нужно уточнение: {question}"
        return reply

    async def cancel_reminder(args: BaseModel) -> str:
        assert isinstance(args, CancelReminderArgs)
        candidates = await service.resolve(args.reminder)
        active = [r for r in candidates if r.active]
        if not active:
            return (
                f"Действующее напоминание по ссылке {args.reminder!r} не найдено. "
                "Посмотри список через list_reminders и используй id."
            )
        if len(active) > 1:
            return _ambiguous(service, active)
        return await service.cancel(active[0])

    return [
        ToolSpec(
            name="create_reminder",
            description="Создать напоминание — само придёт в срок; когда — словами в when",
            args_schema=CreateReminderArgs,
            risk=RiskLevel.WRITE,
            module="reminders",
            handler=create_reminder,
        ),
        ToolSpec(
            name="list_reminders",
            description="Показать предстоящие напоминания",
            args_schema=ListRemindersArgs,
            risk=RiskLevel.READ,
            module="reminders",
            handler=list_reminders,
        ),
        ToolSpec(
            name="snooze_reminder",
            description="Перенести напоминание на другое время",
            args_schema=SnoozeReminderArgs,
            risk=RiskLevel.WRITE,
            module="reminders",
            handler=snooze_reminder,
        ),
        ToolSpec(
            name="cancel_reminder",
            description="Отменить напоминание (у повторяющегося — всю серию)",
            args_schema=CancelReminderArgs,
            risk=RiskLevel.WRITE,
            module="reminders",
            handler=cancel_reminder,
        ),
    ]

"""Инструменты задач: создать / найти / отметить / изменить (FR-4.1).

Результаты размечены как данные: модель обязана сообщать id и сроки только
из ответа инструмента (защита от фабрикации, урок Sprint 2). Ссылка на
задачу — короткий id (#a1b2c3) или слова из названия; при нескольких
совпадениях инструмент возвращает кандидатов и просит уточнить.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sba.core import execctx
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.modules.tasks.service import TasksService
from sba.modules.tasks.store import TaskView

WHEN_DESCRIPTION = (
    "Срок словами: 'завтра в 15:00', 'в пятницу', 'каждый понедельник', "
    "'25.12'. Пусто — без срока"
)


class CreateTaskArgs(BaseModel):
    title: str = Field(description="Суть задачи без даты и без слов 'создай задачу'")
    when: str = Field(default="", description=WHEN_DESCRIPTION)
    project: str = Field(default="", description="Название проекта, если назван")
    notes: str = Field(default="", description="Детали и заметки, если есть")


class SearchTasksArgs(BaseModel):
    query: str = Field(default="", description="Слова для поиска; пусто — все задачи")
    project: str = Field(default="", description="Фильтр по названию проекта")
    status: Literal["open", "done", "cancelled", "all"] = Field(
        default="open", description="open — открытые (по умолчанию), all — любые"
    )


class CompleteTaskArgs(BaseModel):
    task: str = Field(description="Какую задачу отметить сделанной: id вида a1b2c3 "
                      "или несколько слов из названия")


class UpdateTaskArgs(BaseModel):
    task: str = Field(description="Какую задачу изменить: id вида a1b2c3 или слова из названия")
    new_title: str = Field(default="", description="Новое название; пусто — не менять")
    when: str = Field(default="", description="Новый срок словами; 'без срока' — снять срок; "
                      "пусто — не менять")
    project: str = Field(default="", description="Новый проект; пусто — не менять")
    notes: str = Field(default="", description="Новые заметки; пусто — не менять")
    status: Literal["", "open", "cancelled"] = Field(
        default="", description="cancelled — отменить задачу, open — вернуть в работу; "
        "пусто — не менять"
    )


def _ambiguous(service: TasksService, candidates: list[TaskView]) -> str:
    lines = [f"• {service.format_task(t)}" for t in candidates]
    return (
        "Нашлось несколько подходящих задач — уточни по id:\n" + "\n".join(lines)
    )


def build_task_tools(service: TasksService) -> list[ToolSpec]:
    async def create_task(args: BaseModel) -> str:
        assert isinstance(args, CreateTaskArgs)
        outcome = await service.create(
            title=args.title.strip(),
            when=args.when,
            project=args.project,
            notes=args.notes,
            source_message_id=execctx.current_message_id.get(),
        )
        if outcome.task is None:
            return f"❓ Задача НЕ создана, нужно уточнение у пользователя: {outcome.question}"
        text = f"Создал задачу: {service.format_task(outcome.task)}"
        if outcome.warning:
            text += f"\n⚠️ Внимание: {outcome.warning}."
        return text

    async def search_tasks(args: BaseModel) -> str:
        assert isinstance(args, SearchTasksArgs)
        tasks = await service.search(args.query, args.project, args.status)
        if not tasks:
            scope = f" по запросу {args.query!r}" if args.query else ""
            scope += f" в проекте {args.project!r}" if args.project else ""
            return (
                f"Задач{scope} не найдено. Так и скажи пользователю — не выдумывай задачи."
            )
        lines = [f"• {service.format_task(t)}" for t in tasks]
        return "Найденные задачи (перечисляй только их, с id):\n" + "\n".join(lines)

    async def complete_task(args: BaseModel) -> str:
        assert isinstance(args, CompleteTaskArgs)
        candidates = await service.resolve(args.task)
        open_candidates = [t for t in candidates if t.status == "open"]
        if not open_candidates:
            return (
                f"Открытая задача по ссылке {args.task!r} не найдена. "
                "Найди её через search_tasks и используй id."
            )
        if len(open_candidates) > 1:
            return _ambiguous(service, open_candidates)
        updated, next_due = await service.complete(open_candidates[0])
        if next_due is not None:
            return (
                f"✅ Отметил сделанной: {updated.title}. Задача повторяется, "
                f"следующий срок: {next_due}."
            )
        return f"✅ Задача закрыта: {updated.title}."

    async def update_task(args: BaseModel) -> str:
        assert isinstance(args, UpdateTaskArgs)
        candidates = await service.resolve(args.task)
        if not candidates:
            return (
                f"Задача по ссылке {args.task!r} не найдена. "
                "Найди её через search_tasks и используй id."
            )
        if len(candidates) > 1:
            return _ambiguous(service, candidates)
        updated, question, warning = await service.update(
            candidates[0],
            title=args.new_title,
            when=args.when,
            project=args.project,
            notes=args.notes,
            status=args.status,
        )
        if question is not None:
            return f"❓ Задача НЕ изменена, нужно уточнение у пользователя: {question}"
        text = f"Обновил задачу: {service.format_task(updated)}"
        if warning:
            text += f"\n⚠️ Внимание: {warning}."
        return text

    return [
        ToolSpec(
            name="create_task",
            description="Создать задачу; срок и повторение передай словами в when",
            args_schema=CreateTaskArgs,
            risk=RiskLevel.WRITE,
            module="tasks",
            handler=create_task,
        ),
        ToolSpec(
            name="search_tasks",
            description="Найти задачи или показать список (в т.ч. по проекту или статусу)",
            args_schema=SearchTasksArgs,
            risk=RiskLevel.READ,
            module="tasks",
            handler=search_tasks,
        ),
        ToolSpec(
            name="complete_task",
            description="Отметить задачу сделанной (повторяющаяся сдвинется на следующий раз)",
            args_schema=CompleteTaskArgs,
            risk=RiskLevel.WRITE,
            module="tasks",
            handler=complete_task,
        ),
        ToolSpec(
            name="update_task",
            description="Изменить задачу: название, срок, проект, заметки, отмена/возврат",
            args_schema=UpdateTaskArgs,
            risk=RiskLevel.WRITE,
            module="tasks",
            handler=update_task,
        ),
    ]

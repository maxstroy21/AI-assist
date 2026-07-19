"""Task Manager: фасад задач для инструментов и сервис-команды /tasks.

Правила:
- срок разбирает WhenParser; неоднозначность → переспрос (задача НЕ создаётся);
- если разбор срока недоступен (Ollama лежит, роль не настроена) — задача
  создаётся без срока с честным предупреждением: CRUD не зависит от LLM;
- выполнение повторяющейся задачи не закрывает её, а сдвигает срок на
  следующее срабатывание RRULE.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo

import structlog

from sba.core.events import EventBus
from sba.llm.gateway import LLMError
from sba.modules.tasks.dates import WhenParseError, WhenParser, describe_rrule, next_occurrence
from sba.modules.tasks.events import TaskChanged
from sba.modules.tasks.store import TaskStore, TaskView

log = structlog.get_logger(__name__)

# один владелец (допущение A-3), как в MemoryService
OWNER_ID = "owner"

WEEKDAY_SHORT = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")

STATUS_LABELS = {"open": "открыта", "done": "сделана", "cancelled": "отменена"}


@dataclass(frozen=True)
class CreateOutcome:
    task: TaskView | None
    question: str | None = None   # переспрос: срок неоднозначен, задача не создана
    warning: str | None = None    # задача создана без срока (разбор недоступен)


class TasksService:
    def __init__(
        self,
        store: TaskStore,
        parser: WhenParser,
        timezone: tzinfo,
        owner_id: str = OWNER_ID,
        list_limit: int = 15,
        bus: EventBus | None = None,
    ) -> None:
        self._store = store
        self._parser = parser
        self._tz = timezone
        self._owner = owner_id
        self._list_limit = list_limit
        self._bus = bus

    async def _publish_changed(
        self, reason: str, task: TaskView
    ) -> None:
        if self._bus is None:
            return
        await self._bus.publish(
            TaskChanged(
                reason=reason,  # type: ignore[arg-type]
                task_id=task.id,
                title=task.title,
                status=task.status,
                due=task.due,
                rrule=task.rrule,
            )
        )

    # ── операции для инструментов ────────────────────────────────────────────

    async def create(
        self,
        title: str,
        when: str = "",
        project: str = "",
        notes: str = "",
        source_message_id: str | None = None,
    ) -> CreateOutcome:
        due: str | None = None
        rrule: str | None = None
        warning: str | None = None
        try:
            parsed = await self._parser.parse(when)
            if parsed.kind == "unclear":
                return CreateOutcome(task=None, question=parsed.question)
            if parsed.due is not None:
                due = parsed.due.isoformat()
            rrule = parsed.rrule
        except (WhenParseError, LLMError) as exc:
            log.warning("when_parse_failed", when=when, error=str(exc))
            warning = (
                f"не смог разобрать срок «{when}», задача создана без срока — "
                "уточните его позже (например: «перенеси её на завтра»)"
            )
        task = await self._store.add(
            self._owner,
            title=title,
            notes=notes,
            project=project.strip() or None,
            due=due,
            rrule=rrule,
            source_message_id=source_message_id,
        )
        await self._publish_changed("created", task)
        return CreateOutcome(task=task, warning=warning)

    async def search(
        self, query: str = "", project: str = "", status: str = "open",
        limit: int | None = None,
    ) -> list[TaskView]:
        return await self._store.search(
            self._owner,
            query=query,
            project=project.strip() or None,
            status=None if status == "all" else status,
            limit=limit if limit is not None else self._list_limit,
        )

    async def resolve(self, ref: str) -> list[TaskView]:
        return await self._store.resolve(self._owner, ref)

    async def get(self, task_id: str) -> TaskView | None:
        return await self._store.get(self._owner, task_id)

    async def complete(self, task: TaskView) -> tuple[TaskView, str | None]:
        """Закрыть задачу; у повторяющихся — сдвинуть срок. Возвращает
        (обновлённая задача, срок следующего повторения по-русски | None)."""
        now = datetime.now(self._tz)
        stamp = datetime.now(UTC).isoformat()
        if task.rrule and task.due:
            due_dt = datetime.fromisoformat(task.due)
            nxt = next_occurrence(task.rrule, due_dt, max(now, due_dt))
            if nxt is not None:
                updated = await self._store.update(
                    self._owner, task.id,
                    due=nxt.isoformat(), rrule=task.rrule, completed_at=stamp,
                )
                assert updated is not None
                await self._publish_changed("completed", updated)
                return updated, self.format_due(updated.due)
        updated = await self._store.update(
            self._owner, task.id, status="done", completed_at=stamp
        )
        assert updated is not None
        await self._publish_changed("completed", updated)
        return updated, None

    async def update(
        self,
        task: TaskView,
        *,
        title: str = "",
        when: str = "",
        project: str = "",
        notes: str = "",
        status: str = "",
    ) -> tuple[TaskView, str | None, str | None]:
        """Точечное изменение. Возвращает (задача, переспрос, предупреждение);
        при переспросе задача не изменена."""
        due: str | None = None
        rrule: str | None = None
        clear_due = False
        warning: str | None = None
        if when.strip():
            try:
                parsed = await self._parser.parse(when)
                if parsed.kind == "unclear":
                    return task, parsed.question, None
                if parsed.kind == "none":
                    clear_due = True
                else:
                    due = parsed.due.isoformat() if parsed.due is not None else None
                    rrule = parsed.rrule
                    if due is None and rrule is not None:
                        clear_due = True  # rrule без вычислимого срока: записать с пустым due
            except (WhenParseError, LLMError) as exc:
                log.warning("when_parse_failed", when=when, error=str(exc))
                warning = f"не смог разобрать срок «{when}» — срок не изменён"
        updated = await self._store.update(
            self._owner,
            task.id,
            title=title.strip() or None,
            notes=notes.strip() or None,
            project=project if project.strip() else None,
            status=status.strip() or None,
            due=due,
            clear_due=clear_due,
            rrule=rrule,
        )
        assert updated is not None
        await self._publish_changed("updated", updated)
        return updated, None, warning

    async def count_open(self) -> int:
        return await self._store.count_open(self._owner)

    # ── форматирование для пользователя и модели ─────────────────────────────

    def format_due(self, due_iso: str | None) -> str | None:
        if not due_iso:
            return None
        due = datetime.fromisoformat(due_iso).astimezone(self._tz)
        now = datetime.now(self._tz)
        text = f"{WEEKDAY_SHORT[due.weekday()]} {due.day:02d}.{due.month:02d}"
        if due.year != now.year:
            text += f".{due.year}"
        text += f" {due.hour:02d}:{due.minute:02d}"
        return text

    def format_task(self, task: TaskView) -> str:
        parts: list[str] = []
        if task.project:
            parts.append(f"проект: {task.project}")
        due_text = self.format_due(task.due)
        if due_text:
            overdue = (
                task.status == "open"
                and datetime.fromisoformat(task.due) < datetime.now(self._tz)
                if task.due
                else False
            )
            parts.append(f"срок: {due_text}" + (" ⚠️ просрочена" if overdue else ""))
        if task.rrule:
            parts.append(f"повтор: {describe_rrule(task.rrule)}")
        if task.status != "open":
            parts.append(STATUS_LABELS.get(task.status, task.status))
        if task.notes:
            parts.append(f"заметки: {task.notes}")
        suffix = f" ({'; '.join(parts)})" if parts else ""
        return f"#{task.short_id} {task.title}{suffix}"

    async def overview_text(self) -> str:
        """Для /tasks: открытые задачи по срокам, мимо LLM."""
        tasks = await self.search()
        if not tasks:
            return "Открытых задач нет."
        total = await self.count_open()
        lines = [f"Открытые задачи ({total}):"]
        lines += [f"• {self.format_task(t)}" for t in tasks]
        if total > len(tasks):
            lines.append(f"… и ещё {total - len(tasks)} — спросите «что у меня по задачам»")
        return "\n".join(lines)

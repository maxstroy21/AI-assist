"""Инструменты памяти: запомнить / вспомнить / забыть (FR-3.8)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.modules.memory.service import MemoryService, format_fact

FactType = Literal["person", "project", "decision", "preference", "fact"]


class RememberArgs(BaseModel):
    type: FactType = Field(
        description="person — о человеке; project — о проекте; decision — принятое "
        "решение; preference — предпочтение владельца (стиль, вкусы); fact — прочее"
    )
    subject: str = Field(
        description="Кого или чего касается, кратко: 'Иван Петров', 'проект Экспедиция', "
        "'стиль ответов'"
    )
    content: str = Field(description="Само знание, 1–3 предложения")


class RecallArgs(BaseModel):
    query: str = Field(description="О чём вспомнить: имя, тема, вопрос")


class ForgetArgs(BaseModel):
    query: str = Field(description="Что именно забыть: тема или формулировка")


def build_memory_tools(service: MemoryService) -> list[ToolSpec]:
    async def remember_fact(args: BaseModel) -> str:
        assert isinstance(args, RememberArgs)
        fact = await service.remember(args.type, args.subject, args.content)
        return f"Запомнил. {format_fact(fact)}"

    async def recall_memory(args: BaseModel) -> str:
        assert isinstance(args, RecallArgs)
        facts = await service.recall(args.query)
        if not facts:
            return f"В памяти ничего не найдено по запросу {args.query!r}."
        lines = [f"• {format_fact(f)} (записано {f.created_at[:10]})" for f in facts]
        return "Найдено в памяти:\n" + "\n".join(lines)

    async def forget_memory(args: BaseModel) -> str:
        assert isinstance(args, ForgetArgs)
        forgotten = await service.forget(args.query)
        if not forgotten:
            return (
                f"По запросу {args.query!r} ничего не нашлось — нечего забывать. "
                "Уточните формулировку."
            )
        lines = [f"• {format_fact(f)}" for f in forgotten]
        return "Забыто:\n" + "\n".join(lines)

    return [
        ToolSpec(
            name="remember_fact",
            description="Сохранить факт в долговременную память (переживает перезапуски)",
            args_schema=RememberArgs,
            risk=RiskLevel.WRITE,
            module="memory",
            handler=remember_fact,
        ),
        ToolSpec(
            name="recall_memory",
            description="Найти в долговременной памяти факты по теме или имени",
            args_schema=RecallArgs,
            risk=RiskLevel.READ,
            module="memory",
            handler=recall_memory,
        ),
        ToolSpec(
            name="forget_memory",
            description="Забыть факты по теме (мягкое удаление из памяти)",
            args_schema=ForgetArgs,
            risk=RiskLevel.WRITE,
            module="memory",
            handler=forget_memory,
        ),
    ]

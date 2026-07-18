"""Базовые инструменты: время. Первый модуль-пробник агентности."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from sba.core.tools.spec import RiskLevel, ToolSpec

WEEKDAYS_RU = [
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
]


class NoArgs(BaseModel):
    pass


def build_tools(timezone: str) -> list[ToolSpec]:
    tz = ZoneInfo(timezone)

    async def get_current_time(args: BaseModel) -> str:
        now = datetime.now(tz)
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}, {WEEKDAYS_RU[now.weekday()]} ({tz.key})"

    return [
        ToolSpec(
            name="get_current_time",
            description="Текущие дата, время и день недели",
            args_schema=NoArgs,
            risk=RiskLevel.READ,
            module="basic",
            handler=get_current_time,
        )
    ]

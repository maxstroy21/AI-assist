"""Декларация инструмента: то, что модуль отдаёт в Tool Registry.

Трёхуровневая модель риска (ADR-10, docs/02-architecture.md §5.1):
read — свободно; write — свободно с журналом; destructive — только
после явного подтверждения пользователя.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class RiskLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: type[BaseModel]
    risk: RiskLevel
    module: str
    handler: Callable[[BaseModel], Awaitable[str]]
    # Готовая JSON Schema аргументов вместо генерации из args_schema — для
    # MCP-тулов (Sprint 9): внешний сервер уже прислал схему, а args_schema
    # тогда — «пропускающая» модель (валидацию делает сам внешний сервер)
    json_schema: dict[str, Any] | None = None

    def to_openai(self) -> dict[str, Any]:
        if self.json_schema is not None:
            schema = self.json_schema
        else:
            schema = self.args_schema.model_json_schema()
            schema.pop("title", None)
            for prop in schema.get("properties", {}).values():
                if isinstance(prop, dict):
                    prop.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    text: str
    error: bool = False

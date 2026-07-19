"""Tool Registry: регистрация инструментов, валидация аргументов, исполнение
с аудитом и перехватом destructive-вызовов (требуют подтверждения).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection

import structlog
from pydantic import ValidationError

from sba.core.tools.spec import RiskLevel, ToolResult, ToolSpec
from sba.infra.audit import AuditLog
from sba.llm.gateway import ToolCall, ToolSchema

log = structlog.get_logger(__name__)

RESULT_MAX_CHARS = 6000  # защита контекста модели от гигантских результатов
TOOL_TIMEOUT_SECONDS = 90.0  # инструмент не должен вешать agent loop


class ConfirmationRequired(Exception):
    """destructive-инструмент без подтверждения: оркестратор спросит пользователя."""

    def __init__(self, spec: ToolSpec, call: ToolCall) -> None:
        super().__init__(spec.name)
        self.spec = spec
        self.call = call


class ToolRegistry:
    def __init__(self, audit: AuditLog) -> None:
        self._audit = audit
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"инструмент {spec.name!r} уже зарегистрирован")
        self._tools[spec.name] = spec
        log.info("tool_registered", tool=spec.name, module=spec.module, risk=spec.risk)

    def available(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def openai_schemas(self, modules: Collection[str] | None = None) -> list[ToolSchema]:
        """Схемы инструментов для модели. modules — ограничить набор модулями
        (topic-scoped, ускоряет промпт); None — все зарегистрированные."""
        return [
            spec.to_openai()
            for spec in self._tools.values()
            if modules is None or spec.module in modules
        ]

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    async def execute(self, call: ToolCall, confirmed: bool = False) -> ToolResult:
        spec = self._tools.get(call.name)
        args_json = json.dumps(call.arguments, ensure_ascii=False)
        if spec is None:
            await self._audit.record("tool_call", call.name, f"неизвестный инструмент: {args_json}")
            return ToolResult(
                text=f"Ошибка: инструмента {call.name!r} не существует. "
                f"Доступны: {', '.join(self._tools)}",
                error=True,
            )
        if spec.risk == RiskLevel.DESTRUCTIVE and not confirmed:
            await self._audit.record("confirmation_requested", call.name, args_json)
            raise ConfirmationRequired(spec, call)

        await self._audit.record(
            "tool_call",
            call.name,
            args_json,
            confirmed=confirmed if spec.risk == RiskLevel.DESTRUCTIVE else None,
        )
        result = await self._run(spec, call)
        await self._audit.record(
            "tool_result",
            call.name,
            ("ОШИБКА: " if result.error else "") + result.text[:500],
        )
        return result

    async def _run(self, spec: ToolSpec, call: ToolCall) -> ToolResult:
        try:
            args = spec.args_schema.model_validate(call.arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
            )
            return ToolResult(text=f"Ошибка аргументов: {problems}", error=True)
        try:
            text = await asyncio.wait_for(spec.handler(args), timeout=TOOL_TIMEOUT_SECONDS)
        except TimeoutError:
            log.error("tool_timeout", tool=spec.name)
            return ToolResult(
                text=f"Инструмент {spec.name} не уложился в "
                f"{int(TOOL_TIMEOUT_SECONDS)} секунд и был прерван",
                error=True,
            )
        except Exception as exc:  # инструмент не должен ронять agent loop
            log.error("tool_failed", tool=spec.name, error=str(exc))
            return ToolResult(text=f"Ошибка выполнения {spec.name}: {exc}", error=True)
        if len(text) > RESULT_MAX_CHARS:
            text = text[:RESULT_MAX_CHARS] + "\n…(результат обрезан)"
        return ToolResult(text=text)

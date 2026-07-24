"""MCP Client Hub (Sprint 9): внешние MCP-серверы из конфига → Tool Registry.

Каждый сервер из ``mcp.servers`` поднимается в собственной фоновой задаче
(она владеет соединением и переподключается при падении), его инструменты
оборачиваются в ToolSpec и регистрируются в общем реестре: для агента они
неотличимы от внутренних и подчиняются тем же уровням риска (ADR-7, ADR-10).
Риск задаётся конфигом, по умолчанию — консервативно destructive.

Недоверие то же, что к LLM: чужой сервер может висеть или падать, поэтому
подключение — с таймаутом, вызовы — под таймаутом реестра, а сбой одного
сервера не мешает ни старту приложения, ни остальным серверам.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict

from sba.core.tools.registry import ToolRegistry
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.infra.config import McpServerEntry

log = structlog.get_logger(__name__)

RECONNECT_DELAY_SECONDS = 5.0
# ожидание результата вызова: должно быть меньше таймаута реестра (90 с),
# чтобы пользователь получил внятную ошибку хаба, а не общий таймаут
CALL_TIMEOUT_SECONDS = 75.0


class ExternalToolArgs(BaseModel):
    """«Пропускающая» модель аргументов MCP-тула: схему прислал внешний сервер,
    он же и валидирует — мы передаём аргументы как есть."""

    model_config = ConfigDict(extra="allow")


@dataclass
class _CallRequest:
    tool: str
    arguments: dict[str, Any]
    future: asyncio.Future[str]


@dataclass
class _ServerState:
    entry: McpServerEntry
    queue: asyncio.Queue[_CallRequest] = field(default_factory=asyncio.Queue)
    task: asyncio.Task[None] | None = None


class MCPClientHub:
    def __init__(self, servers: list[McpServerEntry]) -> None:
        self._servers = {s.name: _ServerState(entry=s) for s in servers if s.enabled}
        # имена «модулей» зарегистрированных серверов — для topic-scoped фильтра
        self.module_names: set[str] = set()

    # ── запуск и регистрация ─────────────────────────────────────────────────

    async def start(self, registry: ToolRegistry) -> None:
        """Поднять серверы и зарегистрировать их инструменты. Сбой любого
        сервера — предупреждение в лог, не отказ старта приложения."""
        for state in self._servers.values():
            ready: asyncio.Future[list[Any]] = asyncio.get_running_loop().create_future()
            state.task = asyncio.create_task(
                self._worker(state, ready), name=f"mcp-{state.entry.name}"
            )
            try:
                tools = await asyncio.wait_for(
                    ready, timeout=state.entry.connect_timeout_seconds
                )
            except Exception as exc:  # таймаут ожидания или отказ соединения
                state.task.cancel()
                state.task = None
                log.warning(
                    "mcp_server_unavailable",
                    server=state.entry.name,
                    error=str(exc) or type(exc).__name__,
                    hint="проверьте command/url в mcp.servers; сервер пропущен",
                )
                continue
            self._register_tools(registry, state, tools)

    def _register_tools(
        self, registry: ToolRegistry, state: _ServerState, tools: list[Any]
    ) -> None:
        entry = state.entry
        module = f"mcp:{entry.name}"
        for tool in tools:
            name = str(tool.name)
            if registry.get(name) is not None:
                # коллизия с внутренним инструментом или другим сервером —
                # даём префикс, чтобы оба остались доступны
                name = f"{entry.name}_{tool.name}"
                if registry.get(name) is not None:
                    log.warning("mcp_tool_name_conflict", server=entry.name, tool=tool.name)
                    continue
            risk = RiskLevel(entry.tool_risks.get(str(tool.name), entry.risk))
            schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {
                "type": "object", "properties": {},
            }
            registry.register(
                ToolSpec(
                    name=name,
                    description=(tool.description or f"Инструмент MCP-сервера {entry.name}"),
                    args_schema=ExternalToolArgs,
                    risk=risk,
                    module=module,
                    handler=self._make_handler(state, str(tool.name)),
                    json_schema=schema,
                )
            )
        self.module_names.add(module)
        log.info("mcp_server_connected", server=entry.name, tools=len(tools))

    def _make_handler(self, state: _ServerState, tool: str) -> Any:
        async def call(args: BaseModel) -> str:
            request = _CallRequest(
                tool=tool,
                arguments=args.model_dump(exclude_none=True),
                future=asyncio.get_running_loop().create_future(),
            )
            await state.queue.put(request)
            try:
                return await asyncio.wait_for(request.future, timeout=CALL_TIMEOUT_SECONDS)
            except TimeoutError:
                return (
                    f"MCP-сервер {state.entry.name!r} не ответил на вызов {tool!r} "
                    "за отведённое время. Возможно, он завис или переподключается."
                )

        return call

    # ── фоновая задача сервера: соединение живёт здесь ───────────────────────

    async def _worker(
        self, state: _ServerState, ready: asyncio.Future[list[Any]]
    ) -> None:
        """Владелец соединения: контексты транспорта открываются и закрываются
        в ОДНОЙ задаче (требование anyio), падение → переподключение."""
        entry = state.entry
        while True:
            try:
                async with AsyncExitStack() as stack:
                    session = await self._connect(stack, entry)
                    tools = await self._list_all_tools(session)
                    if not ready.done():
                        ready.set_result(tools)
                    await self._serve_calls(state, session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not ready.done():
                    # первый коннект не удался — наружу, start() пропустит сервер
                    ready.set_exception(exc)
                    return
                log.warning(
                    "mcp_server_lost",
                    server=entry.name,
                    error=str(exc),
                    retry_seconds=RECONNECT_DELAY_SECONDS,
                )
                await asyncio.sleep(RECONNECT_DELAY_SECONDS)

    async def _connect(self, stack: AsyncExitStack, entry: McpServerEntry) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        if entry.command:
            params = StdioServerParameters(
                command=entry.command,
                args=entry.args,
                env={**get_default_environment(), **entry.env},
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        else:
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(entry.url)
            )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _list_all_tools(self, session: Any) -> list[Any]:
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                return tools

    async def _serve_calls(self, state: _ServerState, session: Any) -> None:
        """Исполнять вызовы из очереди, пока соединение живо. Ошибка вызова
        отдаётся автору запроса, ошибка транспорта — наружу (переподключение)."""
        while True:
            request = await state.queue.get()
            try:
                result = await session.call_tool(request.tool, request.arguments)
            except Exception as exc:
                if not request.future.done():
                    request.future.set_result(
                        f"Ошибка вызова MCP-инструмента {request.tool!r}: {exc}"
                    )
                raise  # соединение подозрительно — пересоздать
            if not request.future.done():
                request.future.set_result(_result_text(state.entry.name, request.tool, result))

    async def stop(self) -> None:
        for state in self._servers.values():
            if state.task is not None:
                state.task.cancel()
                try:
                    await state.task
                except (asyncio.CancelledError, Exception):
                    pass
                state.task = None


def _result_text(server: str, tool: str, result: Any) -> str:
    """Свести содержимое ответа MCP к тексту для контекста модели."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    if not parts:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            parts.append(json.dumps(structured, ensure_ascii=False))
    text = "\n".join(parts).strip() or "(пустой ответ MCP-сервера)"
    if getattr(result, "isError", False):
        return f"Ошибка MCP-инструмента {tool!r} (сервер {server}): {text}"
    return text

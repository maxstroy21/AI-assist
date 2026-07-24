"""MCP Client Hub (Sprint 9): регистрация внешних тулов, риск, переподключение.

Транспорт подменяется фейковой сессией: протокол MCP тестирует SDK,
нам важен маппинг «внешний тул → ToolSpec в общем реестре».
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from sba.core.tools.registry import ConfirmationRequired, ToolRegistry
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.infra.audit import AuditLog
from sba.infra.config import McpServerEntry
from sba.infra.db import Database
from sba.llm.gateway import ToolCall
from sba.mcp import client_hub
from sba.mcp.client_hub import MCPClientHub

FETCH_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string", "description": "Адрес страницы"}},
    "required": ["url"],
}


def fake_tool(name="fetch_page", schema=FETCH_SCHEMA):
    return SimpleNamespace(name=name, description="Скачать страницу", inputSchema=schema)


class FakeSession:
    def __init__(self, tools, result_text="OK", is_error=False, fail_first_call=False):
        self.tools = tools
        self.result_text = result_text
        self.is_error = is_error
        self.fail_first_call = fail_first_call
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self, cursor=None):
        return SimpleNamespace(tools=self.tools, nextCursor=None)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail_first_call:
            self.fail_first_call = False
            raise RuntimeError("труба лопнула")
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.result_text)],
            structuredContent=None,
            isError=self.is_error,
        )


@pytest.fixture
async def registry(tmp_path):
    db = await Database.open(tmp_path / "sba.db")
    yield ToolRegistry(AuditLog(db))
    await db.close()


def entry(**kw) -> McpServerEntry:
    params = dict(name="fetch", command="uvx", args=["mcp-server-fetch"])
    params.update(kw)
    return McpServerEntry.model_validate(params)


def hub_with_session(server_entry, session) -> MCPClientHub:
    hub = MCPClientHub([server_entry])

    async def fake_connect(stack, e):
        return session

    hub._connect = fake_connect  # type: ignore[method-assign]
    return hub


async def test_tools_registered_with_schema_and_risk(registry):
    hub = hub_with_session(entry(risk="read"), FakeSession([fake_tool()]))
    await hub.start(registry)
    spec = registry.get("fetch_page")
    assert spec is not None
    assert spec.risk == RiskLevel.READ
    assert spec.module == "mcp:fetch"
    assert hub.module_names == {"mcp:fetch"}
    # модель видит схему, которую прислал сервер, а не пустую заглушку
    assert spec.to_openai()["function"]["parameters"] == FETCH_SCHEMA
    await hub.stop()


async def test_default_risk_is_destructive(registry):
    hub = hub_with_session(entry(), FakeSession([fake_tool()]))
    await hub.start(registry)
    with pytest.raises(ConfirmationRequired):
        await registry.execute(
            ToolCall(id="1", name="fetch_page", arguments={"url": "http://x"})
        )
    await hub.stop()


async def test_tool_risk_override(registry):
    hub = hub_with_session(
        entry(tool_risks={"fetch_page": "read"}), FakeSession([fake_tool()])
    )
    await hub.start(registry)
    assert registry.get("fetch_page").risk == RiskLevel.READ
    await hub.stop()


async def test_call_goes_through_registry(registry):
    session = FakeSession([fake_tool()], result_text="страница скачана")
    hub = hub_with_session(entry(risk="read"), session)
    await hub.start(registry)
    result = await registry.execute(
        ToolCall(id="1", name="fetch_page", arguments={"url": "http://x"})
    )
    assert not result.error
    assert result.text == "страница скачана"
    assert session.calls == [("fetch_page", {"url": "http://x"})]
    await hub.stop()


async def test_error_result_marked(registry):
    session = FakeSession([fake_tool()], result_text="404", is_error=True)
    hub = hub_with_session(entry(risk="read"), session)
    await hub.start(registry)
    result = await registry.execute(
        ToolCall(id="1", name="fetch_page", arguments={"url": "http://x"})
    )
    assert "Ошибка MCP-инструмента" in result.text
    assert "404" in result.text
    await hub.stop()


async def test_unavailable_server_skipped(registry):
    hub = MCPClientHub([entry()])

    async def failing_connect(stack, e):
        raise ConnectionError("нет такого процесса")

    hub._connect = failing_connect  # type: ignore[method-assign]
    await hub.start(registry)  # не бросает — приложение стартует без сервера
    assert registry.available() == []
    assert hub.module_names == set()
    await hub.stop()


async def test_name_conflict_gets_prefix(registry):
    from pydantic import BaseModel

    async def internal(args: BaseModel) -> str:
        return "внутренний"

    registry.register(
        ToolSpec(
            name="fetch_page",
            description="внутренний тул",
            args_schema=BaseModel,
            risk=RiskLevel.READ,
            module="basic",
            handler=internal,
        )
    )
    hub = hub_with_session(entry(risk="read"), FakeSession([fake_tool()]))
    await hub.start(registry)
    spec = registry.get("fetch_fetch_page")
    assert spec is not None and spec.module == "mcp:fetch"
    # внутренний не затёрт
    assert registry.get("fetch_page").module == "basic"
    await hub.stop()


async def test_reconnect_after_transport_failure(registry, monkeypatch):
    monkeypatch.setattr(client_hub, "RECONNECT_DELAY_SECONDS", 0.01)
    session = FakeSession([fake_tool()], fail_first_call=True)
    connects = 0
    hub = MCPClientHub([entry(risk="read")])

    async def fake_connect(stack, e):
        nonlocal connects
        connects += 1
        return session

    hub._connect = fake_connect  # type: ignore[method-assign]
    await hub.start(registry)

    first = await registry.execute(
        ToolCall(id="1", name="fetch_page", arguments={"url": "http://x"})
    )
    assert "Ошибка вызова MCP-инструмента" in first.text
    await asyncio.sleep(0.1)  # воркер успевает переподключиться
    second = await registry.execute(
        ToolCall(id="2", name="fetch_page", arguments={"url": "http://x"})
    )
    assert second.text == "OK"
    assert connects >= 2
    await hub.stop()


async def test_disabled_server_ignored(registry):
    hub = MCPClientHub([entry(enabled=False)])
    await hub.start(registry)
    assert registry.available() == []
    await hub.stop()

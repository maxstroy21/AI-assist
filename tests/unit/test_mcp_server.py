"""MCP Server (Sprint 9): экспорт модулей из конфига, работа через MCP-клиент.

Клиент SDK соединяется с нашим сервером через in-memory транспорт —
та же механика, что у Claude Desktop, только без stdio-процесса.
"""

from __future__ import annotations

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from sba.infra.config import Config
from sba.infra.db import Database
from sba.mcp.server import build_export_registry, build_mcp_server


def make_config(tmp_path, modules: list[str]) -> Config:
    return Config.model_validate(
        {
            "app": {"data_dir": str(tmp_path / "data")},
            "mcp": {"export_modules": modules},
        }
    )


@pytest.fixture
async def db(tmp_path):
    database = await Database.open(tmp_path / "data" / "sba.db")
    yield database
    await database.close()


async def test_export_by_module_list(tmp_path, db):
    config = make_config(tmp_path, ["basic", "files", "memory"])
    registry, gateway = build_export_registry(config, tmp_path, db)
    assert gateway is None  # для этих модулей LLM-гейтвей не нужен
    names = {spec.name for spec in registry.available()}
    assert "get_current_time" in names
    assert "list_files" in names
    assert "remember_fact" in names
    # destructive не экспортируется никогда: внешний клиент не умеет подтверждений
    assert "delete_file" not in names
    # неэкспортированный модуль отсутствует
    assert "create_task" not in names


async def test_unknown_module_is_warning_not_crash(tmp_path, db):
    config = make_config(tmp_path, ["привет"])
    registry, _ = build_export_registry(config, tmp_path, db)
    assert registry.available() == []


async def test_tools_visible_and_callable_from_mcp_client(tmp_path, db):
    config = make_config(tmp_path, ["basic", "memory"])
    registry, _ = build_export_registry(config, tmp_path, db)
    server = build_mcp_server(registry)

    async with create_connected_server_and_client_session(server) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed.tools}
        assert {"get_current_time", "remember_fact", "recall_memory"} <= names
        # схема аргументов дошла до клиента
        remember = next(t for t in listed.tools if t.name == "remember_fact")
        assert "subject" in remember.inputSchema.get("properties", {})

        # вызов реального инструмента: память записана и находится
        stored = await client.call_tool(
            "remember_fact",
            {"type": "fact", "subject": "кофе", "content": "Владелец пьёт эспрессо"},
        )
        assert not stored.isError
        recalled = await client.call_tool("recall_memory", {"query": "кофе"})
        assert not recalled.isError
        assert "эспрессо" in recalled.content[0].text


async def test_bad_arguments_reported_as_error(tmp_path, db):
    config = make_config(tmp_path, ["memory"])
    registry, _ = build_export_registry(config, tmp_path, db)
    server = build_mcp_server(registry)

    async with create_connected_server_and_client_session(server) as client:
        result = await client.call_tool("remember_fact", {"type": "fact"})
        assert result.isError


async def test_calls_are_audited(tmp_path, db):
    config = make_config(tmp_path, ["basic"])
    registry, _ = build_export_registry(config, tmp_path, db)
    server = build_mcp_server(registry)

    async with create_connected_server_and_client_session(server) as client:
        await client.call_tool("get_current_time", {})

    rows = await db.fetch_all("SELECT kind, name FROM audit_log ORDER BY id")
    kinds = [(r["kind"], r["name"]) for r in rows]
    assert ("tool_call", "get_current_time") in kinds
    assert ("tool_result", "get_current_time") in kinds

"""MCP Server (Sprint 9): экспорт модулей из конфига, работа через MCP-клиент.

Клиент SDK соединяется с нашим сервером через in-memory транспорт —
та же механика, что у Claude Desktop, только без stdio-процесса.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from sba.infra.config import Config
from sba.infra.db import Database
from sba.mcp.server import build_export_registry, build_mcp_server, resolve_data_dir


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


def test_relative_data_dir_resolved_against_project_root():
    """Урок живой проверки: Claude Desktop запускает сервер из чужой рабочей
    папки, поэтому относительный ./data должен считаться от корня проекта
    (рядом с config), а не от текущего каталога — иначе PermissionError."""
    config = Config.model_validate({"app": {"data_dir": "./data"}})
    config_dir = Path("C:/Users/mmirosh/AI-assist/config")
    resolved = resolve_data_dir(config, config_dir)
    assert resolved.is_absolute()
    assert resolved.name == "data"
    assert resolved.parent.name == "AI-assist"


def test_absolute_data_dir_kept_as_is(tmp_path):
    absolute = tmp_path / "state"
    config = Config.model_validate({"app": {"data_dir": str(absolute)}})
    resolved = resolve_data_dir(config, tmp_path / "config")
    assert resolved == absolute


MINIMAL_MODELS = (
    "runtimes:\n  ollama:\n    kind: openai_compatible\n"
    "    base_url: http://localhost:11434/v1\n    api_key: ''\n"
    "roles:\n"
    "  chat: {runtime: ollama, model: qwen2.5:3b-instruct}\n"
    "  extraction: {runtime: ollama, model: qwen2.5:3b-instruct}\n"
    "  embedding: {runtime: ollama, model: bge-m3}\n"
)


async def test_rag_export_never_touches_embedded_qdrant(tmp_path, db):
    """Урок живой проверки: embedded Qdrant держит основное приложение, второй
    процесс его не откроет (portalocker виснет/падает). Экспортный сервер
    обязан регистрировать search_documents, но НЕ создавать файлы qdrant —
    поиск идёт лексически (FTS по общей SQLite)."""
    config = make_config(tmp_path, ["rag"])
    (tmp_path / "models.yaml").write_text(MINIMAL_MODELS, encoding="utf-8")
    registry, gateway = build_export_registry(config, tmp_path, db)
    names = {spec.name for spec in registry.available()}
    assert "search_documents" in names
    # ключевое: на диске не появилось векторное хранилище — значит его не
    # открывали и не заблокировали бы работающий бот
    assert not (tmp_path / "data" / "qdrant").exists()
    if gateway is not None:
        await gateway.aclose()


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

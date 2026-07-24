"""MCP Server (Sprint 9): экспорт наших инструментов внешним MCP-клиентам.

Отдельный entrypoint (``python -m sba.mcp.server``): Claude Desktop и другие
MCP-клиенты получают доступ к «мозгу» ассистента — поиску по документам,
памяти, задачам. Транспорт — stdio (протокол в stdout, поэтому НИКАКИХ
print: логи идут в stderr через structlog).

Экспортируется не список имён инструментов, а СПИСОК МОДУЛЕЙ из конфига
(``mcp.export_modules``, docs/06 §Sprint 9): новый инструмент существующего
модуля утекает наружу сам собой, смена набора — конфиг, не код.

Ограничения осознанные:
- destructive-инструменты не экспортируются никогда: внешний клиент не
  умеет наших подтверждений «да/нет», а рисковать файлами владельца молча
  нельзя (ADR-10);
- SQLite общий с основным приложением (WAL переживает два процесса), а вот
  embedded Qdrant второй процесс открыть не может — если основное
  приложение запущено, поиск по документам работает лексически (FTS),
  это штатная деградация, не ошибка;
- память здесь всегда в FTS-режиме recall по той же причине.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from sba import __version__
from sba.core.tools.registry import ToolRegistry
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.core.types import new_id
from sba.infra.audit import AuditLog
from sba.infra.config import Config, ConfigError, load_config
from sba.infra.db import Database
from sba.infra.logging import setup_logging
from sba.infra.vectors import VectorStore
from sba.llm.config import load_models_config
from sba.llm.gateway import ToolCall
from sba.llm.service import ModelGateway
from sba.modules.basic.tools import build_tools as build_basic_tools
from sba.modules.files.tools import FilesToolset
from sba.modules.memory.episodes import EpisodeStore
from sba.modules.memory.service import MemoryService
from sba.modules.memory.store import MemoryStore
from sba.modules.memory.tools import build_memory_tools
from sba.modules.rag.service import RAGService
from sba.modules.rag.store import ChunkStore
from sba.modules.rag.tools import build_rag_tools
from sba.modules.tasks.dates import WhenParser
from sba.modules.tasks.service import TasksService
from sba.modules.tasks.store import TaskStore
from sba.modules.tasks.tools import build_task_tools

log = structlog.get_logger(__name__)

SERVER_NAME = "second-brain"


def build_export_registry(
    config: Config, config_dir: Path, db: Database
) -> tuple[ToolRegistry, ModelGateway | None]:
    """Собрать реестр инструментов для модулей из mcp.export_modules.

    Композиция — облегчённая копия app.py: без каналов, шины и фоновых
    работ; создаются только сервисы, чьи модули экспортируются.
    """
    audit = AuditLog(db)
    registry = ToolRegistry(audit)
    tz = ZoneInfo(config.app.timezone)

    gateway: ModelGateway | None = None

    def get_gateway() -> ModelGateway:
        nonlocal gateway
        if gateway is None:
            gateway = ModelGateway(load_models_config(config_dir / "models.yaml"))
        return gateway

    specs: list[ToolSpec] = []
    for module in config.mcp.export_modules:
        if module == "basic":
            specs.extend(build_basic_tools(config.app.timezone))
        elif module == "files":
            specs.extend(FilesToolset(config.files).build_tools())
        elif module == "rag":
            rag_cfg = config.modules.rag
            try:
                vectors = VectorStore.open(config.app.data_dir / "qdrant")
            except Exception as exc:
                # embedded Qdrant уже открыт основным приложением — работаем
                # с пустым векторным стором: поиск деградирует до FTS
                log.warning(
                    "mcp_vectors_busy", error=str(exc), hint="поиск будет лексическим"
                )
                vectors = VectorStore.in_memory()
            rag = RAGService(
                ChunkStore(db), vectors, get_gateway(), embed_batch=rag_cfg.embed_batch
            )
            specs.extend(
                build_rag_tools(
                    rag,
                    top_k=rag_cfg.search_top_k,
                    snippet_chars=rag_cfg.snippet_chars,
                    configured=bool(rag_cfg.sources),
                )
            )
        elif module == "memory":
            memory = MemoryService(MemoryStore(db), episodes=EpisodeStore(db))
            specs.extend(build_memory_tools(memory))
        elif module == "tasks":
            tasks_cfg = config.modules.tasks
            parser = WhenParser(
                get_gateway() if get_gateway().has_role("extraction") else None,
                tz,
                default_hour=tasks_cfg.default_hour,
                clarify_confidence=tasks_cfg.clarify_confidence,
            )
            tasks = TasksService(
                TaskStore(db), parser, tz, list_limit=tasks_cfg.list_limit
            )
            specs.extend(build_task_tools(tasks))
        else:
            log.warning(
                "mcp_export_unknown_module",
                module=module,
                known=["basic", "files", "rag", "memory", "tasks"],
            )

    exported = 0
    for spec in specs:
        if spec.risk == RiskLevel.DESTRUCTIVE:
            log.info("mcp_export_skipped_destructive", tool=spec.name)
            continue
        registry.register(spec)
        exported += 1
    log.info(
        "mcp_export_ready", modules=config.mcp.export_modules, tools=exported
    )
    return registry, gateway


def build_mcp_server(registry: ToolRegistry) -> Any:
    """Обернуть реестр в MCP-сервер (low-level API SDK)."""
    import mcp.types as mcp_types
    from mcp.server.lowlevel import Server

    server = Server(SERVER_NAME, version=__version__)

    @server.list_tools()  # type: ignore[untyped-decorator]
    async def list_tools() -> list[Any]:
        return [
            mcp_types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.to_openai()["function"]["parameters"],
            )
            for spec in registry.available()
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        result = await registry.execute(
            ToolCall(id=new_id(), name=name, arguments=arguments or {})
        )
        if result.error:
            raise ValueError(result.text)  # SDK превратит в isError-ответ клиенту
        return [mcp_types.TextContent(type="text", text=result.text)]

    return server


async def run_server(config_dir: Path) -> None:
    config = load_config(config_dir)
    # формат json: stderr MCP-сервера Claude Desktop пишет в свой лог-файл,
    # структурированные строки там читаются лучше цветных ANSI-кодов
    setup_logging(config.logging.level, "json")
    log.info("mcp_server_starting", version=__version__)

    db = await Database.open(config.app.data_dir / "sba.db")
    gateway: ModelGateway | None = None
    try:
        registry, gateway = build_export_registry(config, config_dir, db)
        server = build_mcp_server(registry)

        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        if gateway is not None:
            await gateway.aclose()
        await db.close()
        log.info("mcp_server_stopped")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sba.mcp.server",
        description="Second Brain Agent — MCP-сервер (экспорт инструментов)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="папка с default.yaml / local.yaml (по умолчанию: ./config)",
    )
    args = parser.parse_args()
    try:
        asyncio.run(run_server(args.config_dir))
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""E2E MCP-сервера: реальный подпроцесс + stdio, как его запускает Claude Desktop.

Проверяет entrypoint целиком: python -m sba.mcp.server, JSON-RPC по stdio
(логи не должны засорять stdout — иначе протокол ломается), список тулов
и живой вызов.
"""

from __future__ import annotations

import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server_over_stdio(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        f"app:\n  data_dir: {tmp_path / 'data'}\nmcp:\n  export_modules: [basic]\n",
        encoding="utf-8",
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sba.mcp.server", "--config-dir", str(config_dir)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            assert "get_current_time" in names

            result = await session.call_tool("get_current_time", {})
            assert not result.isError
            assert result.content and result.content[0].text

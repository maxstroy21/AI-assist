"""E2E web-чата: приложение целиком, реальный uvicorn, живой WebSocket.

Echo-процессор вместо LLM (как в smoke): проверяется транспорт
«браузер → WS → Router → обработчик → WS», а не модель.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import httpx
import websockets

from sba.app import App


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_webchat_end_to_end(tmp_path):
    port = free_port()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                f"app:\n  data_dir: {tmp_path / 'data'}",
                "agent:\n  processor: echo",
                "channels:",
                "  cli:\n    enabled: false",
                f"  web:\n    enabled: true\n    port: {port}",
            ]
        ),
        encoding="utf-8",
    )
    app = await App.create(config_dir)
    run_task = asyncio.ensure_future(app.run())
    try:
        # ждём, пока uvicorn начнёт слушать порт
        for _ in range(200):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                await asyncio.sleep(0.05)
        else:
            raise AssertionError("web-чат не поднялся")

        # страница чата отдаётся
        async with httpx.AsyncClient() as http:
            page = await http.get(f"http://127.0.0.1:{port}/")
            assert page.status_code == 200
            assert "Second Brain Agent" in page.text

        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            history = json.loads(await ws.recv())
            assert history["type"] == "history"

            await ws.send(json.dumps({"type": "message", "text": "привет, веб"}))
            reply = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            assert reply["type"] == "message"
            assert reply["text"] == "[echo] привет, веб"

        # вторая вкладка видит историю первой (общая сессия канала)
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ws:
            history = json.loads(await ws.recv())
            texts = [m["text"] for m in history["messages"]]
            assert "привет, веб" in texts
            assert "[echo] привет, веб" in texts
    finally:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await run_task

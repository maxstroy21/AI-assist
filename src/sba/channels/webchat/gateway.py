"""Local Web Chat (Sprint 9): FastAPI + WebSocket на машине владельца.

Тонкий адаптер, как и остальные каналы: браузер ⇄ JSON по WebSocket ⇄ Router.
Никакой своей логики диалога — streaming, кнопки и подтверждения приходят
из ядра теми же OutgoingMessage, что и в Telegram (docs/04 §14).

Безопасность: аутентификации нет, поэтому слушаем только localhost
(channels.web.host по умолчанию 127.0.0.1); смена хоста — осознанное
решение владельца (например, Tailscale-интерфейс без публичного IP).

Протокол WS (JSON-объекты):
  клиент → сервер: {"type": "message", "text": …} | {"type": "action", "id": …}
  сервер → клиент: history | message | stream_start | delta | stream_end
                   | action_result  (см. _broadcast-вызовы ниже)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from sba.core.types import IncomingMessage, OutgoingMessage

log = structlog.get_logger(__name__)

INDEX_PATH = Path(__file__).parent / "static" / "index.html"

# история для нового подключения: [{"role": "user"|"assistant", "text": …}];
# реализацию даёт app.py (каналы не лезут в БД напрямую — только Router и типы)
HistoryProvider = Callable[[], Awaitable[list[dict[str, str]]]]


class WebChatChannel:
    name = "web"

    def __init__(
        self,
        host: str,
        port: int,
        handle_incoming: Callable[[IncomingMessage], Awaitable[None]],
        handle_action: Callable[[str, str, str], Awaitable[str]],
        fetch_history: HistoryProvider,
        user_id: str = "local",
    ) -> None:
        self._host = host
        self._port = port
        self._handle_incoming = handle_incoming
        self._handle_action = handle_action
        self._fetch_history = fetch_history
        self._user_id = user_id
        self._clients: set[WebSocket] = set()
        # диалог последовательный: пока модель отвечает, следующее сообщение ждёт
        # (иначе два потока ответов перемешались бы в одной сессии)
        self._dialog_lock = asyncio.Lock()
        self._pending: set[asyncio.Task[None]] = set()
        self._server: object | None = None  # uvicorn.Server (ленивый импорт в start)
        self._index_html = INDEX_PATH.read_text(encoding="utf-8")
        self._app = self._build_app()

    # ── FastAPI-приложение ───────────────────────────────────────────────────

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Second Brain Agent — web chat")

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(self._index_html)

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await self._serve_client(ws)

        return app

    async def _serve_client(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        log.info("web_client_connected", clients=len(self._clients))
        try:
            history = await self._fetch_history()
            await ws.send_text(_dumps({"type": "history", "messages": history}))
            while True:
                raw = await ws.receive_text()
                await self._on_client_payload(ws, raw)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # соединение не должно ронять канал
            log.warning("web_client_error", error=str(exc))
        finally:
            self._clients.discard(ws)
            log.info("web_client_disconnected", clients=len(self._clients))

    async def _on_client_payload(self, ws: WebSocket, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("web_bad_payload", raw=raw[:200])
            return
        kind = payload.get("type")
        if kind == "message":
            text = str(payload.get("text", "")).strip()
            if text:
                # обработка в фоне: цикл чтения WS остаётся живым (кнопки,
                # следующие сообщения), а порядок диалога держит _dialog_lock
                self._spawn(self._process_message(text))
        elif kind == "action":
            action_id = str(payload.get("id", ""))
            if action_id:
                self._spawn(self._process_action(action_id))
        else:
            log.warning("web_unknown_type", type=kind)

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _process_message(self, text: str) -> None:
        async with self._dialog_lock:
            await self._handle_incoming(
                IncomingMessage(user_id=self._user_id, channel=self.name, text=text)
            )

    async def _process_action(self, action_id: str) -> None:
        ack = await self._handle_action(self._user_id, self.name, action_id)
        await self._broadcast(
            {"type": "action_result", "action_id": action_id, "text": ack}
        )

    # ── исходящие (контракт ChannelAdapter / StreamingChannel) ───────────────

    async def send(self, out: OutgoingMessage) -> None:
        await self._broadcast(
            {
                "type": "message",
                "msg_id": out.id,
                "kind": out.kind.value,
                "text": out.text,
                "actions": [{"id": a.id, "label": a.label} for a in out.actions],
            }
        )

    async def send_stream(self, out: OutgoingMessage, deltas: AsyncIterator[str]) -> str:
        await self._broadcast({"type": "stream_start", "msg_id": out.id})
        parts: list[str] = []
        async for delta in deltas:
            parts.append(delta)
            await self._broadcast({"type": "delta", "msg_id": out.id, "text": delta})
        text = "".join(parts)
        await self._broadcast({"type": "stream_end", "msg_id": out.id, "text": text})
        return text

    async def _broadcast(self, payload: dict[str, object]) -> None:
        """Отправить всем открытым вкладкам; мёртвые соединения отбрасываются.
        Нет ни одной вкладки — сообщение просто не показано (история в БД,
        напоминания дублируются в остальные активные каналы)."""
        data = _dumps(payload)
        for ws in list(self._clients):
            try:
                await ws.send_text(data)
            except Exception:
                self._clients.discard(ws)

    # ── жизненный цикл ───────────────────────────────────────────────────────

    async def start(self) -> None:
        import uvicorn  # ленивый импорт: тесты канала не поднимают сервер

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_config=None,   # логи у нас свои (structlog), uvicorn молчит
            access_log=False,
        )
        server = uvicorn.Server(config)
        self._server = server
        log.info("web_chat_started", url=f"http://{self._host}:{self._port}")
        print(f"🌐 Web-чат: http://{self._host}:{self._port}", flush=True)
        await server.serve()

    async def stop(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True  # type: ignore[attr-defined]
        for task in list(self._pending):
            task.cancel()


def _dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)

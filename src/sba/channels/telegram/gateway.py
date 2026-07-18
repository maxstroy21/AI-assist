"""Telegram Gateway (aiogram 3, long polling).

Тонкий адаптер: whitelist → нормализация → Router; ответы — потоково через
периодическое редактирование сообщения, длинные тексты режутся под лимит.
Входящих соединений нет — только исходящий long polling (Local First).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from time import monotonic

import structlog
from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest, TelegramRetryAfter
from aiogram.types import Message

from sba.channels.telegram.split import cut_once, split_text
from sba.core.types import IncomingMessage, OutgoingMessage

log = structlog.get_logger(__name__)

EDIT_INTERVAL_SECONDS = 2.0   # чаще редактировать нельзя — flood limit Telegram
STREAM_SOFT_LIMIT = 3500      # порог, после которого начинаем новое сообщение
TYPING_CURSOR = " ▌"
HEARTBEAT_SECONDS = 15.0      # на CPU модель может думать минуты — показываем, что живы


async def with_heartbeat(
    deltas: AsyncIterator[str], interval: float = HEARTBEAT_SECONDS
) -> AsyncIterator[tuple[str, str]]:
    """Оборачивает поток кусков текста: ('text', кусок) либо ('wait', 'N') —
    сколько секунд подряд поток молчит (модель думает).

    Важно: ожидание без отмены anext() — отмена по таймауту убила бы
    сам генератор-источник.
    """
    iterator = aiter(deltas)
    waited = 0.0
    pending: asyncio.Task[str] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                waited += interval
                yield ("wait", str(int(waited)))
                continue
            task, pending = pending, None
            try:
                delta = task.result()
            except StopAsyncIteration:
                return
            waited = 0.0
            yield ("text", delta)
    finally:
        if pending is not None:
            pending.cancel()


class TelegramChannel:
    name = "telegram"

    def __init__(
        self,
        token: str,
        allowed_user_ids: list[int],
        handle_incoming: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        self._bot = Bot(token=token)
        self._dp = Dispatcher()
        self._allowed = set(allowed_user_ids)
        self._handle_incoming = handle_incoming
        self._dp.message.register(self._on_text, F.text)
        self._dp.message.register(self._on_unsupported)

    # ── жизненный цикл ───────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self._allowed:
            log.warning(
                "telegram_whitelist_empty",
                hint="задайте channels.telegram.allowed_user_ids в local.yaml; "
                "свой id увидите в логе telegram_unauthorized, написав боту",
            )
        me = await self._bot.get_me()
        log.info("telegram_started", bot=me.username)
        await self._dp.start_polling(self._bot, handle_signals=False)

    async def stop(self) -> None:
        with suppress(Exception):
            result = self._dp.stop_polling()
            if inspect.isawaitable(result):
                await result
        await self._bot.session.close()

    # ── входящие ─────────────────────────────────────────────────────────────

    def _authorized(self, message: Message) -> bool:
        user = message.from_user
        if user is not None and user.id in self._allowed:
            return True
        log.warning(
            "telegram_unauthorized",
            user_id=user.id if user else None,
            username=user.username if user else None,
        )
        return False

    async def _on_text(self, message: Message) -> None:
        if not self._authorized(message):
            return  # чужим — молчание (FR-9.2)
        with suppress(Exception):
            await self._bot.send_chat_action(message.chat.id, "typing")
        await self._handle_incoming(
            IncomingMessage(
                user_id=str(message.from_user.id),  # type: ignore[union-attr]
                channel=self.name,
                text=message.text or "",
            )
        )

    async def _on_unsupported(self, message: Message) -> None:
        if not self._authorized(message):
            return
        await message.answer(
            "Пока я понимаю только текст. Голосовые сообщения и файлы подключу "
            "в следующих версиях."
        )

    # ── исходящие ────────────────────────────────────────────────────────────

    async def send(self, out: OutgoingMessage) -> None:
        chat_id = int(out.user_id)
        for part in split_text(out.text) or ["(пусто)"]:
            await self._send_with_flood_control(chat_id, part)

    async def send_stream(self, out: OutgoingMessage, deltas: AsyncIterator[str]) -> str:
        """Черновик «✍️ …» редактируется по мере генерации; длинный ответ
        продолжается новыми сообщениями. Возвращает полный текст ответа."""
        chat_id = int(out.user_id)
        draft = await self._send_with_flood_control(chat_id, "✍️ …")
        full: list[str] = []
        buffer = ""
        shown = ""
        last_edit = monotonic()

        async for kind, payload in with_heartbeat(deltas):
            if kind == "wait":
                base = buffer.strip() or "✍️ …"
                await self._safe_edit(draft, f"{base}\n⏳ модель думает… ({payload} с)")
                shown = ""  # следующий текстовый кусок перерисует черновик
                continue
            delta = payload
            full.append(delta)
            buffer += delta
            if len(buffer) > STREAM_SOFT_LIMIT:
                head, buffer = cut_once(buffer, STREAM_SOFT_LIMIT)
                await self._safe_edit(draft, head)
                draft = await self._send_with_flood_control(chat_id, "✍️ …")
                shown = ""
                last_edit = monotonic()
            elif buffer.strip() and monotonic() - last_edit >= EDIT_INTERVAL_SECONDS:
                if buffer != shown:
                    await self._safe_edit(draft, buffer + TYPING_CURSOR)
                    shown = buffer
                last_edit = monotonic()

        await self._safe_edit(draft, buffer.strip() or "(пустой ответ)")
        return "".join(full)

    async def _safe_edit(self, message: Message, text: str) -> None:
        for attempt in (1, 2):
            try:
                await message.edit_text(text[:4096])
                return
            except TelegramRetryAfter as exc:
                if attempt == 2:
                    return
                await asyncio.sleep(exc.retry_after)
            except TelegramBadRequest:
                return  # «message is not modified» и подобное — не критично

    async def _send_with_flood_control(self, chat_id: int, text: str) -> Message:
        try:
            return await self._bot.send_message(chat_id, text)
        except TelegramRetryAfter as exc:
            await asyncio.sleep(exc.retry_after)
            return await self._bot.send_message(chat_id, text)

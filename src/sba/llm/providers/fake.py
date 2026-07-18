"""Фейк-LLM для тестов: детерминированный сценарий, запись всех вызовов.

Элемент сценария — либо текст ответа, либо список ToolCall (модель «решила»
позвать инструменты). Используется во всех спринтах для тестов агентских
сценариев без реальной модели (NFR-8).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sba.llm.gateway import ChatMessage, ChatResult, Role, StreamEvent, ToolCall, ToolSchema

ScriptItem = str | list[ToolCall]


class FakeLLM:
    def __init__(self, replies: list[ScriptItem] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[tuple[Role, list[ChatMessage]]] = []
        self.seen_tools: list[list[ToolSchema] | None] = []

    def _next_reply(self) -> ScriptItem:
        return self._replies.pop(0) if self._replies else "ok"

    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult:
        self.calls.append((role, list(messages)))
        item = self._next_reply()
        text = item if isinstance(item, str) else ""
        return ChatResult(text=text, model="fake")

    async def stream(
        self,
        role: Role,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append((role, list(messages)))
        self.seen_tools.append(tools)
        item = self._next_reply()
        if isinstance(item, list):
            yield StreamEvent(tool_calls=item)
            return
        # отдаём по словам, чтобы тесты проверяли настоящую потоковую сборку
        for i, word in enumerate(item.split(" ")):
            yield StreamEvent(text=word if i == 0 else " " + word)

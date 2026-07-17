"""Фейк-LLM для тестов: детерминированные ответы, запись всех вызовов.

Используется во всех спринтах, чтобы тестировать агентские сценарии
без реальной модели (NFR-8).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sba.llm.gateway import ChatMessage, ChatResult, Role


class FakeLLM:
    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[tuple[Role, list[ChatMessage]]] = []

    def _next_reply(self) -> str:
        return self._replies.pop(0) if self._replies else "ok"

    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult:
        self.calls.append((role, list(messages)))
        return ChatResult(text=self._next_reply(), model="fake")

    async def stream(self, role: Role, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.calls.append((role, list(messages)))
        text = self._next_reply()
        # отдаём по словам, чтобы тесты проверяли настоящую потоковую сборку
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word

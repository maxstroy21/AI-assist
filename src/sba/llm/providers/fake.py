"""Фейк-LLM для тестов: детерминированные ответы, запись всех вызовов.

Используется во всех спринтах, чтобы тестировать агентские сценарии
без реальной модели (NFR-8).
"""

from __future__ import annotations

from sba.llm.gateway import ChatMessage, ChatResult, Role


class FakeLLM:
    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[tuple[Role, list[ChatMessage]]] = []

    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult:
        self.calls.append((role, list(messages)))
        text = self._replies.pop(0) if self._replies else "ok"
        return ChatResult(text=text, model="fake")

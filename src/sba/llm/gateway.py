"""Контракт LLM Gateway — единственная точка обращения к моделям.

Роли (chat/extraction/...) назначаются в config/models.yaml; остальной код
пишется против этого интерфейса, а не против конкретного рантайма.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import BaseModel

Role = Literal["chat", "extraction", "summarize", "embedding", "rerank", "stt"]

# OpenAI-формат описания инструмента: {"type": "function", "function": {...}}
ToolSchema = dict[str, Any]


class LLMError(Exception):
    """Модель недоступна или вернула некорректный ответ."""


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None  # assistant: какие инструменты позвал
    tool_call_id: str | None = None           # tool: ответ на какой вызов


class ChatResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StreamEvent(BaseModel):
    """Событие потока: кусок текста и/или итоговое решение позвать инструменты."""

    text: str = ""
    tool_calls: list[ToolCall] | None = None


class LLMGateway(Protocol):
    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult: ...

    def stream(
        self,
        role: Role,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Потоковая генерация; при наличии tools модель может решить их позвать."""
        ...

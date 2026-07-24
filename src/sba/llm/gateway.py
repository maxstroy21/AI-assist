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
        tool_choice: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Потоковая генерация; tools — модель может их позвать;
        tool_choice='required' — обязана (если рантайм поддерживает)."""
        ...

    def chat_runtime_is_local(self) -> bool:
        """Локальный ли рантайм у роли chat (для подсказок об ошибках:
        локальная модель — про Ollama, облачная — про сеть/ключ/лимит)."""
        ...


class Embedder(Protocol):
    """Узкий порт для модулей, которым нужны только эмбеддинги."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class Provider(Protocol):
    """Рантайм-провайдер (Ollama, Anthropic и т.п.): единый контракт для
    ModelGateway. Роль → (провайдер, модель, параметры) назначается в models.yaml."""

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult: ...

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...

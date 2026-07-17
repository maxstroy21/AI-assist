"""Контракт LLM Gateway — единственная точка обращения к моделям.

Роли (chat/extraction/...) назначаются в config/models.yaml; остальной код
пишется против этого интерфейса, а не против конкретного рантайма.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from pydantic import BaseModel

Role = Literal["chat", "extraction", "summarize", "embedding", "rerank", "stt"]


class LLMError(Exception):
    """Модель недоступна или вернула некорректный ответ."""


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMGateway(Protocol):
    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult: ...

    def stream(self, role: Role, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Потоковая генерация: выдаёт куски текста по мере появления."""
        ...

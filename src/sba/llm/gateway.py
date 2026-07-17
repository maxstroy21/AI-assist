"""Контракт LLM Gateway (реализация с реальными провайдерами — Sprint 1).

Единственная точка обращения к моделям; роли (chat/extraction/...) назначаются
в config/models.yaml. В Sprint 0 контракт нужен, чтобы тесты и будущий
оркестратор писались против интерфейса, а не против Ollama.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

Role = Literal["chat", "extraction", "summarize", "embedding", "rerank", "stt"]


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

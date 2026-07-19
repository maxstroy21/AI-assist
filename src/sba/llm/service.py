"""Реализация LLM Gateway: роль → (рантайм, модель, параметры) из models.yaml.

Для tool_mode=json включается fallback: инструменты описываются в промпте,
ответ буферизуется и разбирается как возможный JSON-вызов (llm/toolcalling.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from sba.llm.config import ModelsConfig, RoleConfig
from sba.llm.gateway import (
    ChatMessage,
    ChatResult,
    LLMError,
    Role,
    StreamEvent,
    ToolSchema,
)
from sba.llm.providers.openai_compat import OpenAICompatProvider
from sba.llm.toolcalling import inject_tools_instruction, parse_tool_call_text

log = structlog.get_logger(__name__)


class ModelGateway:
    def __init__(self, config: ModelsConfig) -> None:
        self._config = config
        self._providers: dict[str, OpenAICompatProvider] = {
            name: OpenAICompatProvider(rt.base_url, rt.api_key)
            for name, rt in config.runtimes.items()
        }

    def _resolve(self, role: Role) -> tuple[OpenAICompatProvider, RoleConfig]:
        role_config = self._config.roles.get(role)
        if role_config is None:
            raise LLMError(f"роль {role!r} не настроена в models.yaml")
        return self._providers[role_config.runtime], role_config

    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult:
        provider, rc = self._resolve(role)
        return await provider.chat(rc.model, messages, rc.temperature)

    async def stream(
        self,
        role: Role,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        provider, rc = self._resolve(role)
        if not tools or rc.tool_mode == "native":
            async for event in provider.stream(
                rc.model, messages, rc.temperature, tools=tools, tool_choice=tool_choice
            ):
                yield event
            return
        # json-fallback: буферизуем ответ целиком и решаем, вызов это или текст
        prepared = inject_tools_instruction(messages, tools)
        parts: list[str] = []
        async for event in provider.stream(rc.model, prepared, rc.temperature):
            parts.append(event.text)
        text = "".join(parts)
        calls = parse_tool_call_text(text)
        if calls:
            yield StreamEvent(tool_calls=calls)
        else:
            yield StreamEvent(text=text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Векторы для роли embedding (порт Embedder для RAG и памяти)."""
        provider, rc = self._resolve("embedding")
        return await provider.embed(rc.model, texts)

    def has_role(self, role: Role) -> bool:
        return role in self._config.roles

    async def warmup(self) -> None:
        """Держит chat-модель загруженной в RAM: запрос на 1 токен сбрасывает
        таймер выгрузки Ollama (OLLAMA_KEEP_ALIVE, по умолчанию 5 минут).
        Без этого первый вопрос после простоя ждёт холодную загрузку минутами."""
        try:
            provider, rc = self._resolve("chat")
            await provider.chat(
                rc.model,
                [ChatMessage(role="user", content="ping")],
                temperature=0.0,
                max_tokens=1,
            )
            log.debug("llm_keep_warm_ok", model=rc.model)
        except Exception as exc:  # прогрев не должен ничего ронять
            log.warning("llm_keep_warm_failed", error=str(exc))

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()

"""Реализация LLM Gateway: роль → (рантайм, модель, параметры) из models.yaml.

Для tool_mode=json включается fallback: инструменты описываются в промпте,
ответ буферизуется и разбирается как возможный JSON-вызов (llm/toolcalling.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from urllib.parse import urlparse

import structlog

from sba.llm.config import ModelsConfig, RoleConfig, RuntimeConfig
from sba.llm.gateway import (
    ChatMessage,
    ChatResult,
    LLMError,
    Provider,
    Role,
    StreamEvent,
    ToolSchema,
)
from sba.llm.providers.anthropic_native import AnthropicProvider
from sba.llm.providers.openai_compat import OpenAICompatProvider
from sba.llm.toolcalling import inject_tools_instruction, parse_tool_call_text

log = structlog.get_logger(__name__)


def _build_provider(rt: RuntimeConfig) -> Provider:
    if rt.kind == "anthropic":
        return AnthropicProvider(rt.api_key)
    return OpenAICompatProvider(rt.base_url, rt.api_key)


class ModelGateway:
    def __init__(self, config: ModelsConfig) -> None:
        self._config = config
        self._providers: dict[str, Provider] = {
            name: _build_provider(rt) for name, rt in config.runtimes.items()
        }

    def _resolve(self, role: Role) -> tuple[Provider, RoleConfig]:
        role_config = self._config.roles.get(role)
        if role_config is None:
            raise LLMError(f"роль {role!r} не настроена в models.yaml")
        return self._providers[role_config.runtime], role_config

    async def chat(self, role: Role, messages: list[ChatMessage]) -> ChatResult:
        provider, rc = self._resolve(role)
        return await provider.chat(rc.model, messages, rc.temperature, max_tokens=rc.max_tokens)

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
                rc.model, messages, rc.temperature, tools=tools,
                tool_choice=tool_choice, max_tokens=rc.max_tokens,
            ):
                yield event
            return
        # json-fallback: буферизуем ответ целиком и решаем, вызов это или текст
        prepared = inject_tools_instruction(messages, tools)
        parts: list[str] = []
        async for event in provider.stream(
            rc.model, prepared, rc.temperature, max_tokens=rc.max_tokens
        ):
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

    def chat_runtime_is_local(self) -> bool:
        """Прогрев имеет смысл только для локальной модели (она живёт в RAM этой
        машины). Облаку прогрев не нужен, а на бесплатных тарифах (Groq) пинг
        раз в N минут ещё и съедает дневной лимит запросов."""
        rc = self._config.roles.get("chat")
        if rc is None:
            return False
        rt = self._config.runtimes[rc.runtime]
        if rt.kind == "anthropic":
            return False
        host = urlparse(rt.base_url).hostname or ""
        return host in ("localhost", "127.0.0.1", "::1")

    async def warmup(self) -> bool:
        """Держит chat-модель загруженной в RAM: запрос на 1 токен сбрасывает
        таймер выгрузки Ollama (OLLAMA_KEEP_ALIVE, по умолчанию 5 минут).
        Без этого первый вопрос после простоя ждёт холодную загрузку минутами.

        Возвращает True, если модель ответила (значит, она в памяти). Ошибку
        не бросает — прогрев не должен ронять приложение; False = не дождались
        (холодная загрузка на CPU может превышать таймаут чтения)."""
        try:
            provider, rc = self._resolve("chat")
            await provider.chat(
                rc.model,
                [ChatMessage(role="user", content="ping")],
                temperature=0.0,
                max_tokens=1,
            )
            log.debug("llm_keep_warm_ok", model=rc.model)
            return True
        except Exception as exc:  # прогрев не должен ничего ронять
            log.warning("llm_keep_warm_failed", error=str(exc))
            return False

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()

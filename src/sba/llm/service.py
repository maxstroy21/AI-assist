"""Реализация LLM Gateway: роль → (рантайм, модель, параметры) из models.yaml."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sba.llm.config import ModelsConfig, RoleConfig
from sba.llm.gateway import ChatMessage, ChatResult, LLMError, Role
from sba.llm.providers.openai_compat import OpenAICompatProvider


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

    async def stream(self, role: Role, messages: list[ChatMessage]) -> AsyncIterator[str]:
        provider, rc = self._resolve(role)
        async for delta in provider.stream(rc.model, messages, rc.temperature):
            yield delta

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()

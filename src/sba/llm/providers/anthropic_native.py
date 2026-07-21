"""Провайдер Anthropic (облачный Claude) через официальный SDK `anthropic`.

Роль назначается в models.yaml (`kind: anthropic`). Авторизация — стандартная
для SDK: переменная `ANTHROPIC_API_KEY`, либо вход под аккаунтом владельца через
`ant auth login` (профиль на диске, SDK подхватывает его сам). Ключ в models.yaml
писать НЕ нужно — файл в git; пустой `api_key` = SDK берёт учётку из окружения.

Внутренний формат сообщений у нас OpenAI-образный (ChatMessage с tool_calls);
здесь он переводится в Messages API Anthropic:
- системные сообщения → верхнеуровневый параметр `system`;
- вызовы инструментов ассистента → блоки `tool_use`, ответы инструментов →
  блоки `tool_result` в сообщении роли user;
- схемы инструментов `{function:…}` → `{name, description, input_schema}`.

Температуру не отправляем: новые модели Claude (Opus 4.8 / Sonnet 5) её не
принимают. Эмбеддингов у Anthropic нет — роль `embedding` должна жить на Ollama.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic
import structlog

from sba.llm.gateway import ChatMessage, ChatResult, LLMError, StreamEvent, ToolCall, ToolSchema

log = structlog.get_logger(__name__)

DEFAULT_MAX_TOKENS = 4096
# сеть до облака быстрая, но ответ модели с рассуждениями бывает долгим
TIMEOUT_SECONDS = 300.0


def _split_system(messages: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Anthropic держит системный промпт отдельно. Собираем все system-сообщения
    (у нас их несколько: базовый промпт + факты памяти + нуджи) в один текст."""
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    rest = [m for m in messages if m.role != "system"]
    return "\n\n".join(system_parts), rest


def _to_anthropic_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Перевод истории в формат Anthropic. Соседние сообщения одной роли
    сливаются (ответы инструментов — в один user-ход; чередование ролей не рвётся)."""
    result: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": blocks})

    for msg in messages:
        if msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content.strip():
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.tool_calls or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            append("assistant", blocks)
        elif msg.role == "tool":
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                ],
            )
        else:  # user
            if msg.content:
                append("user", [{"type": "text", "text": msg.content}])
    return result


def _to_anthropic_tools(tools: list[ToolSchema] | None) -> list[dict[str, Any]]:
    """OpenAI-схема {type:function, function:{name, description, parameters}} →
    Anthropic {name, description, input_schema}."""
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function", {})
        converted.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def _to_anthropic_tool_choice(tool_choice: str | None) -> dict[str, str] | None:
    if tool_choice == "required":
        return {"type": "any"}  # Anthropic реально принуждает (в отличие от Ollama)
    return None  # auto по умолчанию


class AnthropicProvider:
    def __init__(self, api_key: str = "", client: anthropic.AsyncAnthropic | None = None) -> None:
        # пустой ключ → None: SDK сам разрешит учётку (ANTHROPIC_API_KEY либо
        # профиль `ant auth login`). Явно "" привёл бы к попытке с пустым ключом
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key or None, timeout=TIMEOUT_SECONDS
        )

    async def aclose(self) -> None:
        await self._client.close()

    def _params(
        self,
        model: str,
        messages: list[ChatMessage],
        max_tokens: int | None,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        system, rest = _split_system(messages)
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "messages": _to_anthropic_messages(rest),
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = _to_anthropic_tools(tools)
            choice = _to_anthropic_tool_choice(tool_choice)
            if choice is not None:
                params["tool_choice"] = choice
        return params

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        params = self._params(model, messages, max_tokens)
        try:
            response = await self._client.messages.create(**params)
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic: {exc}") from exc
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        usage = response.usage
        return ChatResult(
            text=text,
            model=str(response.model),
            prompt_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        params = self._params(model, messages, max_tokens, tools=tools, tool_choice=tool_choice)
        # накопление вызовов инструментов по индексу блока: id/name приходят в
        # content_block_start, аргументы — по кускам JSON в input_json_delta
        partial: dict[int, dict[str, str]] = {}
        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    kind = getattr(event, "type", "")
                    if kind == "content_block_start":
                        block = event.content_block
                        if getattr(block, "type", "") == "tool_use":
                            partial[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "json": "",
                            }
                    elif kind == "content_block_delta":
                        delta = event.delta
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            if delta.text:
                                yield StreamEvent(text=delta.text)
                        elif dtype == "input_json_delta":
                            slot = partial.get(event.index)
                            if slot is not None:
                                slot["json"] += delta.partial_json or ""
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic: {exc}") from exc
        calls = self._finalize_calls(partial)
        if calls:
            yield StreamEvent(tool_calls=calls)

    @staticmethod
    def _finalize_calls(partial: dict[int, dict[str, str]]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(partial):
            slot = partial[index]
            raw = slot["json"].strip() or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"инструмент {slot['name']}: аргументы не JSON: {raw[:200]}"
                ) from exc
            if not isinstance(arguments, dict):
                raise LLMError(f"инструмент {slot['name']}: аргументы не объект")
            calls.append(ToolCall(id=slot["id"], name=slot["name"], arguments=arguments))
        return calls

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        raise LLMError(
            "у Anthropic нет API эмбеддингов — оставьте роль 'embedding' на Ollama "
            "(bge-m3) в config/models.yaml, либо отключите семантический поиск"
        )

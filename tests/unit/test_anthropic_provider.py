"""Тесты нативного провайдера Anthropic: перевод форматов + разбор стрима."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sba.llm.gateway import ChatMessage, LLMError, ToolCall
from sba.llm.providers.anthropic_native import (
    AnthropicProvider,
    _split_system,
    _to_anthropic_messages,
    _to_anthropic_tool_choice,
    _to_anthropic_tools,
)

# ── перевод сообщений ────────────────────────────────────────────────────────


def test_system_messages_are_hoisted() -> None:
    msgs = [
        ChatMessage(role="system", content="ты ассистент"),
        ChatMessage(role="system", content="отвечай кратко"),
        ChatMessage(role="user", content="привет"),
    ]
    system, rest = _split_system(msgs)
    assert system == "ты ассистент\n\nотвечай кратко"
    assert [m.role for m in rest] == ["user"]


def test_tool_call_and_result_translate_to_blocks() -> None:
    msgs = [
        ChatMessage(role="user", content="сколько времени"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="get_time", arguments={"tz": "MSK"})],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="12:00"),
    ]
    result = _to_anthropic_messages(msgs)
    assert result[0] == {"role": "user", "content": [{"type": "text", "text": "сколько времени"}]}
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "get_time", "input": {"tz": "MSK"}}
    ]
    assert result[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "12:00"}],
    }


def test_adjacent_same_role_messages_merge() -> None:
    # два ответа инструментов подряд должны слиться в один user-ход
    msgs = [
        ChatMessage(role="tool", tool_call_id="a", content="раз"),
        ChatMessage(role="tool", tool_call_id="b", content="два"),
    ]
    result = _to_anthropic_messages(msgs)
    assert len(result) == 1
    assert len(result[0]["content"]) == 2


def test_tools_schema_translation() -> None:
    openai_tool = {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "список файлов",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
    converted = _to_anthropic_tools([openai_tool])
    assert converted == [
        {
            "name": "list_files",
            "description": "список файлов",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_tool_choice_required_becomes_any() -> None:
    assert _to_anthropic_tool_choice("required") == {"type": "any"}
    assert _to_anthropic_tool_choice(None) is None


# ── фейковый SDK-клиент ──────────────────────────────────────────────────────


class _FakeMessages:
    def __init__(self, response: Any = None, events: list[Any] | None = None) -> None:
        self._response = response
        self._events = events or []
        self.last_params: dict[str, Any] = {}

    async def create(self, **params: Any) -> Any:
        self.last_params = params
        return self._response

    def stream(self, **params: Any):
        self.last_params = params
        events = self._events

        class _Ctx:
            async def __aenter__(self_inner):
                async def gen():
                    for event in events:
                        yield event

                return gen()

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()


class _FakeClient:
    def __init__(self, response: Any = None, events: list[Any] | None = None) -> None:
        self.messages = _FakeMessages(response, events)

    async def close(self) -> None:
        pass


# ── chat / stream / embed ────────────────────────────────────────────────────


async def test_chat_extracts_text_and_usage() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="здравствуйте")],
        model="claude-sonnet-5",
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )
    client = _FakeClient(response=response)
    provider = AnthropicProvider(client=client)  # type: ignore[arg-type]
    result = await provider.chat("claude-sonnet-5", [ChatMessage(role="user", content="hi")])
    assert result.text == "здравствуйте"
    assert result.prompt_tokens == 7
    assert result.completion_tokens == 3
    # max_tokens обязателен для Anthropic — провайдер подставляет дефолт
    assert client.messages.last_params["max_tokens"] == 4096


async def test_stream_yields_text_then_tool_calls() -> None:
    events = [
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="text_delta", text="Ищу")),
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="text_delta", text="…")),
        SimpleNamespace(type="content_block_start", index=1,
                        content_block=SimpleNamespace(type="tool_use", id="c9",
                                                      name="find_files")),
        SimpleNamespace(type="content_block_delta", index=1,
                        delta=SimpleNamespace(type="input_json_delta",
                                              partial_json='{"name_pattern":')),
        SimpleNamespace(type="content_block_delta", index=1,
                        delta=SimpleNamespace(type="input_json_delta",
                                              partial_json='"*.pdf"}')),
    ]
    provider = AnthropicProvider(client=_FakeClient(events=events))  # type: ignore[arg-type]
    collected = [ev async for ev in provider.stream("m", [ChatMessage(role="user", content="q")])]
    texts = "".join(ev.text for ev in collected if ev.text)
    assert texts == "Ищу…"
    calls = [c for ev in collected if ev.tool_calls for c in ev.tool_calls]
    assert len(calls) == 1
    assert calls[0].name == "find_files"
    assert calls[0].arguments == {"name_pattern": "*.pdf"}


async def test_embed_is_unsupported() -> None:
    provider = AnthropicProvider(client=_FakeClient())  # type: ignore[arg-type]
    with pytest.raises(LLMError, match="эмбеддинг"):
        await provider.embed("m", ["текст"])

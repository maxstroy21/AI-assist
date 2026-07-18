"""JSON-fallback для моделей без нативного tool calling (tool_mode: json).

Инструменты описываются текстом в system prompt; ответ модели, похожий на
{"tool": ..., "arguments": {...}}, превращается в ToolCall. Это страховка
(П-2 из docs/01-requirements.md): работает с любой моделью, ценой потери
потоковости на ход с инструментом.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sba.llm.gateway import ToolCall, ToolSchema

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)


def render_tools_instruction(tools: list[ToolSchema]) -> str:
    lines = [
        "",
        "ДОСТУПНЫЕ ИНСТРУМЕНТЫ:",
    ]
    for tool in tools:
        fn = tool["function"]
        params = json.dumps(fn.get("parameters", {}).get("properties", {}), ensure_ascii=False)
        lines.append(f"- {fn['name']}: {fn.get('description', '')}; аргументы: {params}")
    lines += [
        "",
        "Если для ответа нужен инструмент — ответь ТОЛЬКО одним JSON-объектом вида",
        '{"tool": "имя_инструмента", "arguments": {...}} без пояснений до или после.',
        "Если инструмент не нужен — отвечай обычным текстом.",
    ]
    return "\n".join(lines)


def inject_tools_instruction(
    messages: list[Any], tools: list[ToolSchema]
) -> list[Any]:
    """Копия messages с инструкцией об инструментах в system-сообщении."""
    from sba.llm.gateway import ChatMessage

    instruction = render_tools_instruction(tools)
    result: list[ChatMessage] = []
    injected = False
    for msg in messages:
        if msg.role == "system" and not injected:
            result.append(ChatMessage(role="system", content=msg.content + instruction))
            injected = True
        elif msg.role == "tool":
            # модели без tool calling не знают роли tool — подаём как user
            result.append(
                ChatMessage(role="user", content=f"Результат инструмента:\n{msg.content}")
            )
        elif msg.role == "assistant" and msg.tool_calls:
            calls = json.dumps(
                [{"tool": c.name, "arguments": c.arguments} for c in msg.tool_calls],
                ensure_ascii=False,
            )
            result.append(ChatMessage(role="assistant", content=msg.content or calls))
        else:
            result.append(ChatMessage(role=msg.role, content=msg.content))
    if not injected:
        result.insert(0, ChatMessage(role="system", content=instruction))
    return result


def parse_tool_call_text(text: str) -> list[ToolCall] | None:
    """Ответ модели → ToolCall, если он выглядит как вызов инструмента."""
    cleaned = _CODE_FENCE.sub("", text.strip()).strip()
    candidate = cleaned if cleaned.startswith("{") else None
    if candidate is None:
        match = _JSON_BLOCK.search(cleaned)
        # JSON внутри пояснительного текста принимаем только если он в начале
        if match is None or match.start() > 20:
            return None
        candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name")
    if not isinstance(name, str):
        return None
    arguments = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(arguments, dict):
        return None
    return [ToolCall(id="call_json_0", name=name, arguments=arguments)]

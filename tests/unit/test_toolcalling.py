from sba.llm.gateway import ChatMessage, ToolCall
from sba.llm.toolcalling import (
    inject_tools_instruction,
    parse_tool_call_text,
    render_tools_instruction,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Показать содержимое папки",
            "parameters": {"properties": {"path": {"type": "string"}}},
        },
    }
]


def test_render_mentions_tool_and_format() -> None:
    text = render_tools_instruction(TOOLS)
    assert "list_files" in text
    assert '"tool"' in text


def test_parse_plain_json() -> None:
    calls = parse_tool_call_text('{"tool": "list_files", "arguments": {"path": "."}}')
    assert calls is not None
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "."}


def test_parse_fenced_json() -> None:
    text = '```json\n{"tool": "list_files", "arguments": {"path": "docs"}}\n```'
    calls = parse_tool_call_text(text)
    assert calls is not None
    assert calls[0].arguments == {"path": "docs"}


def test_plain_text_is_not_a_call() -> None:
    assert parse_tool_call_text("Просто отвечаю текстом, без инструментов.") is None


def test_json_buried_in_long_text_is_not_a_call() -> None:
    text = "Вот пример того, как мог бы выглядеть вызов: " * 3 + '{"tool": "x", "arguments": {}}'
    assert parse_tool_call_text(text) is None


def test_inject_appends_to_system_and_converts_tool_roles() -> None:
    messages = [
        ChatMessage(role="system", content="базовый промпт"),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="list_files", arguments={"path": "."})],
        ),
        ChatMessage(role="tool", tool_call_id="1", content="файл.txt"),
    ]
    prepared = inject_tools_instruction(messages, TOOLS)
    assert prepared[0].role == "system"
    assert "базовый промпт" in prepared[0].content
    assert "list_files" in prepared[0].content
    assert prepared[1].role == "assistant"          # tool_calls → текстовое описание
    assert prepared[2].role == "user"               # tool-роль недоступна таким моделям
    assert "файл.txt" in prepared[2].content

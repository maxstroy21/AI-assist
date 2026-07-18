from sba.core.agent.context import build_messages, clean_for_model
from sba.core.history import HistoryEntry


def test_service_lines_stripped_from_assistant_history() -> None:
    content = (
        '🔧 delete_file({"path": "x"})\n'
        "🛑 Действие требует подтверждения\n"
        "Вот содержимое файла:\nстрока данных"
    )
    assert clean_for_model(content) == "Вот содержимое файла:\nстрока данных"


def test_marker_only_assistant_entry_dropped_entirely() -> None:
    history = [
        HistoryEntry("user", "удали файл"),
        HistoryEntry("assistant", "🔧 find_files({})\n⏳ модель думает… (15 с)"),
        HistoryEntry("user", "ну что там?"),
    ]
    messages = build_messages("system", history, budget_chars=1000)
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "user"]  # пустой assistant выпал


def test_user_messages_never_cleaned() -> None:
    history = [HistoryEntry("user", "🔧 что значит этот значок?")]
    messages = build_messages("system", history, budget_chars=1000)
    assert messages[-1].content == "🔧 что значит этот значок?"

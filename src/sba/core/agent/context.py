"""Context Builder v1: system prompt + история в жёстком бюджете символов.

CPU-only железо делает длинный промпт главным врагом латентности
(docs/01-requirements.md §4.2): бюджеты режутся с дальнего конца истории,
самое свежее сообщение не отбрасывается никогда.
"""

from __future__ import annotations

from sba.core.history import HistoryEntry
from sba.llm.gateway import ChatMessage

# Служебные строки интерфейса (маркеры вызовов, подтверждения, статусы) не должны
# попадать в контекст модели: увидев их в истории, она учится их имитировать —
# вплоть до поддельных «🔧 delete_file(...) Успешно удалено» (реальный инцидент).
SERVICE_LINE_PREFIXES = ("🔧", "🛑", "⏳", "🚫", "🆕", "⚠️")


def clean_for_model(content: str) -> str:
    lines = [
        line
        for line in content.splitlines()
        if not line.strip().startswith(SERVICE_LINE_PREFIXES)
    ]
    return "\n".join(lines).strip()


def build_messages(
    system_prompt: str, history: list[HistoryEntry], budget_chars: int
) -> list[ChatMessage]:
    cleaned: list[HistoryEntry] = []
    for entry in history:
        content = entry.content if entry.role == "user" else clean_for_model(entry.content)
        if content:
            cleaned.append(HistoryEntry(role=entry.role, content=content))

    selected: list[HistoryEntry] = []
    used = 0
    for entry in reversed(cleaned):
        cost = len(entry.content)
        if selected and used + cost > budget_chars:
            break
        selected.append(entry)
        used += cost

    messages = [ChatMessage(role="system", content=system_prompt)]
    for entry in reversed(selected):
        if entry.role == "user":
            messages.append(ChatMessage(role="user", content=entry.content))
        else:
            messages.append(ChatMessage(role="assistant", content=entry.content))
    return messages

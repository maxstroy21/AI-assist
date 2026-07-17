"""Context Builder v1: system prompt + история в жёстком бюджете символов.

CPU-only железо делает длинный промпт главным врагом латентности
(docs/01-requirements.md §4.2): бюджеты режутся с дальнего конца истории,
самое свежее сообщение не отбрасывается никогда.
"""

from __future__ import annotations

from sba.core.history import HistoryEntry
from sba.llm.gateway import ChatMessage


def build_messages(
    system_prompt: str, history: list[HistoryEntry], budget_chars: int
) -> list[ChatMessage]:
    selected: list[HistoryEntry] = []
    used = 0
    for entry in reversed(history):
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

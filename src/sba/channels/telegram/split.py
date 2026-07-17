"""Разбиение длинных текстов под лимит Telegram (4096 символов на сообщение)."""

from __future__ import annotations

TELEGRAM_LIMIT = 4096


def cut_once(text: str, limit: int) -> tuple[str, str]:
    """Отрезает от начала кусок ≤ limit по границе абзаца/строки/слова."""
    if len(text) <= limit:
        return text, ""
    for separator in ("\n\n", "\n", " "):
        cut = text.rfind(separator, 0, limit + 1)
        if cut >= limit // 2:
            return text[:cut].rstrip(), text[cut:].lstrip()
    return text[:limit], text[limit:]


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    parts: list[str] = []
    rest = text
    while rest:
        head, rest = cut_once(rest, limit)
        if head:
            parts.append(head)
        elif not rest:
            break
    return parts

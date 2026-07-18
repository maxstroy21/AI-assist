"""Memory Service: фасад памяти для инструментов и Context Builder.

Реализует core-протокол MemoryPort (инверсия зависимостей: ядро не знает
о модуле памяти, модуль реализует порт ядра).
"""

from __future__ import annotations

from sba.modules.memory.store import FactView, MemoryStore

TYPE_LABELS = {
    "person": "человек",
    "project": "проект",
    "decision": "решение",
    "preference": "предпочтение",
    "fact": "факт",
}

PREFERENCES_BUDGET_CHARS = 400
FACTS_BUDGET_CHARS = 600

# один владелец (допущение A-3); мульти-пользовательность добавит ExecContext
OWNER_ID = "owner"


def format_fact(fact: FactView) -> str:
    label = TYPE_LABELS.get(fact.type, fact.type)
    return f"[{label}] {fact.subject}: {fact.content}"


class MemoryService:
    def __init__(self, store: MemoryStore, owner_id: str = OWNER_ID) -> None:
        self._store = store
        self._owner = owner_id

    # ── для инструментов ─────────────────────────────────────────────────────

    async def remember(self, fact_type: str, subject: str, content: str) -> FactView:
        return await self._store.add(self._owner, fact_type, subject, content)

    async def recall(self, query: str, k: int = 6) -> list[FactView]:
        return await self._store.search(self._owner, query, k)

    async def forget(self, query: str) -> list[FactView]:
        return await self._store.retract(self._owner, query)

    async def count(self) -> int:
        return await self._store.count_active(self._owner)

    # ── MemoryPort (для Context Builder) ─────────────────────────────────────

    async def preferences_text(self) -> str | None:
        preferences = await self._store.preferences(self._owner)
        if not preferences:
            return None
        lines: list[str] = []
        used = 0
        for pref in preferences:
            line = f"- {pref.subject}: {pref.content}"
            if used + len(line) > PREFERENCES_BUDGET_CHARS:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines) if lines else None

    async def relevant_facts_text(self, query: str) -> str | None:
        facts = await self._store.search(self._owner, query, k=5)
        relevant = [f for f in facts if f.type != "preference"]
        if not relevant:
            return None
        lines: list[str] = []
        used = 0
        for fact in relevant:
            line = f"- {format_fact(fact)}"
            if used + len(line) > FACTS_BUDGET_CHARS:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines) if lines else None

"""Memory Service: фасад памяти для инструментов и Context Builder.

Реализует core-протокол MemoryPort (инверсия зависимостей: ядро не знает
о модуле памяти, модуль реализует порт ядра). С Sprint 7 отвечает и за
агрегированный ответ «что ты знаешь о N»: факты по типам + история
вытесненных решений + эпизоды прошлых разговоров.
"""

from __future__ import annotations

from sba.modules.memory.episodes import EpisodeStore, EpisodeView
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
    tag = f" [проект: {fact.project}]" if fact.project else ""
    return f"[{label}]{tag} {fact.subject}: {fact.content}"


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        episodes: EpisodeStore | None = None,
        owner_id: str = OWNER_ID,
    ) -> None:
        self._store = store
        self._episodes = episodes
        self._owner = owner_id

    # ── для инструментов ─────────────────────────────────────────────────────

    async def remember(self, fact_type: str, subject: str, content: str) -> FactView:
        return await self._store.add(self._owner, fact_type, subject, content)

    async def recall(
        self, query: str, k: int = 6, project: str | None = None
    ) -> list[FactView]:
        return await self._store.search(self._owner, query, k, project=project)

    async def decision_history(self, fact: FactView) -> list[FactView]:
        """Вытесненные предшественники решения («ранее решали иначе»)."""
        if fact.type not in ("decision", "preference"):
            return []
        return await self._store.history(fact.id)

    async def episodes_about(self, query: str, k: int = 2) -> list[EpisodeView]:
        """Прошлые разговоры по теме (пусто, если эпизоды не подключены)."""
        if self._episodes is None:
            return []
        return await self._episodes.search(self._owner, query, k)

    async def forget(self, query: str) -> list[FactView]:
        return await self._store.retract(self._owner, query)

    async def count(self) -> int:
        return await self._store.count_active(self._owner)

    # ── ручная ревизия (/memory, мимо LLM) ───────────────────────────────────

    async def review_text(self, days: int = 7) -> str:
        """Что запомнено за неделю: страховка от накопления мусора (риск
        Sprint 7) — владелец видит автозаписи и может сказать «забудь …»."""
        total = await self._store.count_active(self._owner)
        recent = await self._store.recent(self._owner, days=days)
        lines = [f"Память: активных фактов — {total}."]
        if self._episodes is not None:
            episodes = await self._episodes.recent(self._owner, days=days)
            lines[0] += f" Эпизодов разговоров за {days} дн. — {len(episodes)}."
        if not recent:
            lines.append(f"За последние {days} дн. новых фактов не появилось.")
            return "\n".join(lines)
        lines.append(f"Новое за {days} дн. (🤖 — записано автоматически):")
        for fact in recent:
            marker = "🤖" if fact.source.startswith("auto:") else "✍️"
            lines.append(f"• {marker} {format_fact(fact)} ({fact.created_at[:10]})")
        lines.append("Убрать лишнее: скажите «забудь …» с темой факта.")
        return "\n".join(lines)

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

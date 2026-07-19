"""Консолидация памяти: закрытый разговор → эпизод → факты (Sprint 7).

Продолжение линии недоверия к модели (уроки Sprint 2/4/5):

- модель НЕ решает, что «важно» бесконтрольно: ответы проходят строгую
  JSON-валидацию, типы и длины ограничены, предпочтения из автоизвлечения
  запрещены (менять поведение ассистента может только явное «запомни»);
- порог уверенности (min_confidence) и потолок фактов с разговора
  (max_facts_per_session) не дают мусору копиться;
- дедупликация детерминированная (пересечение слов), без LLM;
- противоречия не затирают историю: решения по той же теме вытесняются
  через superseded_by (store.add);
- всё внешнее — с таймаутом; сбой откладывает разговор на повтор
  (до MAX_ATTEMPTS), а не роняет цикл.

Отступление от плана «ночной» консолидации (06-roadmap Sprint 7):
ноутбук владельца ночью выключен — ночной джоб с misfire=skip не выполнялся
бы никогда. Вместо этого воркер работает непрерывно в паузах диалога
(cooldown, как индексатор): закрытая сессия консолидируется в ближайшую
тихую минуту, а «осиротевшие» разговоры (закрыть было некому — приложение
перезапустили) добираются периодическим сканом.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from sba.core.agent.context import clean_for_model
from sba.core.events import SessionClosed
from sba.infra.config import MemoryConfig
from sba.infra.db import Database
from sba.llm.gateway import ChatMessage, LLMGateway, Role
from sba.modules.memory.episodes import EpisodeStore
from sba.modules.memory.store import SUPERSEDING_TYPES, MemoryStore

log = structlog.get_logger(__name__)

MAX_ATTEMPTS = 3          # сбойный разговор откладывается, после — оставляем failed
BATCH_LIMIT = 3           # разговоров за один заход (не занимать CPU надолго)
MIN_USER_MESSAGES = 2     # короче — нечего консолидировать («привет» — «привет»)
MIN_USER_CHARS = 60
TRANSCRIPT_BUDGET_CHARS = 6000   # бюджет диалога в промпте (CPU-only, короткие промпты)
TRANSCRIPT_MAX_MESSAGES = 60
ORPHAN_IDLE_HOURS = 3.0   # открытый разговор без активности дольше — считаем брошенным
EXTRACTED_MAX_CONFIDENCE = 0.9   # извлечённое не бывает достовернее явного «запомни»

ALLOWED_FACT_TYPES = frozenset({"person", "project", "decision", "fact"})

_EPISODE_PROMPT = """Ниже — разговор владельца с его ассистентом.
Составь краткий итог для дневника разговоров.

Ответь ОДНИМ JSON-объектом без пояснений и без markdown:
{{"summary": "о чём говорили и к чему пришли, 2-4 предложения по-русски", \
"topics": ["тема 1", "тема 2"], "project": null}}

Правила: пиши только то, что есть в разговоре, ничего не добавляй; topics —
до 5 коротких тем; project — название проекта, если разговор явно про него, \
иначе null.

РАЗГОВОР:
{transcript}"""

_FACTS_PROMPT = """Ниже — разговор владельца с его ассистентом.
Выпиши устойчивые факты, которые ВЛАДЕЛЕЦ сообщил О СЕБЕ, о людях, \
о проектах или о принятых решениях.

Ответь ОДНИМ JSON-объектом без пояснений и без markdown:
{{"facts": [{{"type": "person", "subject": "Иван Петров", \
"content": "подрядчик по смете экспедиции", "project": null, "confidence": 0.8}}]}}

Правила:
- type: person — о человеке; project — о проекте; decision — принятое решение; \
fact — прочее устойчивое знание.
- Бери ТОЛЬКО то, что владелец сказал сам. Слова ассистента фактами не считаются.
- НЕ выписывай: разовые действия и просьбы (создание задач, напоминаний, поиск \
файлов), мнения ассистента, пересказ документов, догадки.
- content — 1-2 коротких предложения; subject — кого или чего касается.
- confidence — твоя уверенность 0..1: 0.9 — владелец сказал прямо; \
0.6 — упомянул вскользь; сомневаешься — не включай факт вовсе.
- Фактов может не быть — тогда {{"facts": []}}. Не больше {max_facts}.

РАЗГОВОР:
{transcript}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_WORD_RE = re.compile(r"\w+")


class ConsolidationError(Exception):
    """Шаг консолидации не удался: LLM недоступна или ответ невалиден."""


def _extract_json(text: str) -> dict[str, Any]:
    m = _JSON_RE.search(text)
    if m is None:
        raise ConsolidationError(f"в ответе модели нет JSON: {text[:120]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ConsolidationError(f"невалидный JSON от модели: {exc}") from exc
    if not isinstance(data, dict):
        raise ConsolidationError("модель вернула не JSON-объект")
    return data


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3}


def content_overlap(a: str, b: str) -> float:
    """Коэффициент пересечения слов (0..1): грубая, но детерминированная
    мера «это то же самое знание» для дедупликации без LLM."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


DUPLICATE_OVERLAP = 0.7


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class ExtractedFact:
    type: str
    subject: str
    content: str
    project: str | None
    confidence: float


def _validate_episode(data: dict[str, Any]) -> tuple[str, list[str], str | None]:
    summary = data.get("summary")
    if not isinstance(summary, str) or not 20 <= len(summary.strip()) <= 600:
        raise ConsolidationError(f"невалидный summary эпизода: {str(summary)[:80]!r}")
    raw_topics = data.get("topics")
    topics: list[str] = []
    if isinstance(raw_topics, list):
        topics = [t.strip() for t in raw_topics if isinstance(t, str) and t.strip()][:5]
        topics = [t[:40] for t in topics]
    project = data.get("project")
    if not isinstance(project, str) or not project.strip():
        project = None
    else:
        project = project.strip()[:60]
    return summary.strip(), topics, project


def _validate_facts(
    data: dict[str, Any], max_facts: int, min_confidence: float
) -> list[ExtractedFact]:
    raw = data.get("facts")
    if raw is None:
        raise ConsolidationError("в ответе модели нет поля facts")
    if not isinstance(raw, list):
        raise ConsolidationError("facts — не список")
    facts: list[ExtractedFact] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fact_type = item.get("type")
        subject = item.get("subject")
        content = item.get("content")
        # предпочтения из автоизвлечения запрещены: поведение ассистента
        # меняет только явное «запомни» (консервативность, риск спринта)
        if fact_type not in ALLOWED_FACT_TYPES:
            continue
        if not isinstance(subject, str) or not 2 <= len(subject.strip()) <= 60:
            continue
        if not isinstance(content, str) or not 5 <= len(content.strip()) <= 300:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if confidence < min_confidence:
            continue
        project = item.get("project")
        if not isinstance(project, str) or not project.strip():
            project = None
        else:
            project = project.strip()[:60]
        facts.append(
            ExtractedFact(
                type=fact_type,
                subject=" ".join(subject.split()),
                content=" ".join(content.split()),
                project=project,
                confidence=min(confidence, EXTRACTED_MAX_CONFIDENCE),
            )
        )
        if len(facts) >= max_facts:
            break
    return facts


class MemoryConsolidator:
    """Фоновый воркер: находит закрытые разговоры и сворачивает их в память."""

    def __init__(
        self,
        db: Database,
        store: MemoryStore,
        episodes: EpisodeStore,
        llm: LLMGateway,
        config: MemoryConfig,
        owner_id: str = "owner",
    ) -> None:
        self._db = db
        self._store = store
        self._episodes = episodes
        self._llm = llm
        self._config = config
        # факты и эпизоды пишутся владельцу (допущение A-3: один пользователь),
        # какой бы канал ни породил разговор (telegram id, cli)
        self._owner = owner_id
        self._last_activity = 0.0
        self._wake = asyncio.Event()

    # ── связь с диалогом ─────────────────────────────────────────────────────

    def notice_activity(self) -> None:
        """Каждое входящее сообщение (подписка на шину в app.py)."""
        self._last_activity = time.monotonic()

    async def on_session_closed(self, event: SessionClosed) -> None:
        """Закрытие сессии — сигнал воркеру проверить очередь пораньше."""
        self._wake.set()

    async def _wait_quiet(self) -> None:
        while True:
            since = time.monotonic() - self._last_activity
            remaining = self._config.dialog_cooldown_seconds - since
            if remaining <= 0:
                return
            await asyncio.sleep(min(remaining, 5.0))

    # ── цикл ─────────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._wake.wait(), timeout=self._config.check_interval_seconds
                )
            except TimeoutError:
                pass
            self._wake.clear()
            try:
                await self._wait_quiet()
                processed = await self.process_pending()
                if processed:
                    log.info("memory_consolidated", conversations=processed)
            except Exception as exc:  # цикл не должен умирать от одного сбоя
                log.error("memory_consolidation_tick_failed", error=str(exc))

    async def process_pending(self, limit: int = BATCH_LIMIT) -> int:
        """Обработать до `limit` неконсолидированных разговоров; вернуть число."""
        processed = 0
        for row in await self._pending(limit):
            await self._process_one(row)
            processed += 1
        return processed

    async def _pending(self, limit: int) -> list[Any]:
        """Закрытые разговоры без записи о консолидации (или с неисчерпанными
        повторами), плюс «осиротевшие» — открытые, но давно брошенные."""
        orphan_cutoff = (
            datetime.now(UTC) - timedelta(hours=ORPHAN_IDLE_HOURS)
        ).isoformat()
        return await self._db.fetch_all(
            "SELECT c.*, m.attempts FROM conversations c"
            " LEFT JOIN memory_consolidations m ON m.conversation_id = c.id"
            " WHERE (c.closed_at IS NOT NULL OR c.last_activity_at < ?)"
            "   AND (m.conversation_id IS NULL"
            "        OR (m.result = 'failed' AND m.attempts < ?))"
            " ORDER BY c.last_activity_at LIMIT ?",
            (orphan_cutoff, MAX_ATTEMPTS, limit),
        )

    # ── обработка одного разговора ───────────────────────────────────────────

    async def _process_one(self, conv: Any) -> None:
        conversation_id = str(conv["id"])
        transcript, user_messages, user_chars = await self._transcript(conversation_id)
        if user_messages < MIN_USER_MESSAGES or user_chars < MIN_USER_CHARS:
            await self._record(conversation_id, "skipped_short", 0)
            log.debug("memory_conversation_skipped_short", conversation=conversation_id)
            return
        try:
            async with asyncio.timeout(self._config.llm_timeout_seconds):
                episode_data = _extract_json(
                    await self._ask("summarize", _EPISODE_PROMPT.format(transcript=transcript))
                )
            summary, topics, project = _validate_episode(episode_data)
            async with asyncio.timeout(self._config.llm_timeout_seconds):
                facts_data = _extract_json(
                    await self._ask(
                        "extraction",
                        _FACTS_PROMPT.format(
                            transcript=transcript,
                            max_facts=self._config.max_facts_per_session,
                        ),
                    )
                )
            facts = _validate_facts(
                facts_data,
                self._config.max_facts_per_session,
                self._config.min_confidence,
            )
        except Exception as exc:
            # ConsolidationError, таймаут, LLMError, сеть — разговор уйдёт на повтор
            await self._record_failure(conversation_id, str(exc))
            return

        closed_at = str(conv["closed_at"] or conv["last_activity_at"])
        await self._episodes.add(
            self._owner,
            conversation_id,
            summary,
            topics,
            project,
            started_at=str(conv["started_at"]),
            closed_at=closed_at,
        )
        added = 0
        for fact in facts:
            if await self._is_duplicate(fact):
                log.debug("memory_fact_deduplicated", subject=fact.subject)
                continue
            await self._store.add(
                self._owner,
                fact.type,
                fact.subject,
                fact.content,
                source=f"auto:{conversation_id[:8]}",
                confidence=fact.confidence,
                project=fact.project or project,
            )
            added += 1
        await self._record(conversation_id, "episode", added)
        log.info(
            "memory_conversation_consolidated",
            conversation=conversation_id,
            facts_added=added,
            topics=topics,
        )

    async def _ask(self, role: Role, prompt: str) -> str:
        result = await self._llm.chat(role, [ChatMessage(role="user", content=prompt)])
        return result.text

    async def _transcript(self, conversation_id: str) -> tuple[str, int, int]:
        """Диалог → текст для промпта. Служебные строки (🔧, ⚠️…) вычищаются —
        модель не должна пересказывать интерфейсные маркеры как факты."""
        rows = await self._db.fetch_all(
            "SELECT role, content FROM messages WHERE conversation_id=?"
            " ORDER BY rowid DESC LIMIT ?",
            (conversation_id, TRANSCRIPT_MAX_MESSAGES),
        )
        lines: list[str] = []
        user_messages = 0
        user_chars = 0
        for row in reversed(rows):
            role, content = str(row["role"]), str(row["content"])
            cleaned = content if role == "user" else clean_for_model(content)
            if not cleaned:
                continue
            if role == "user":
                user_messages += 1
                user_chars += len(cleaned)
            speaker = "Владелец" if role == "user" else "Ассистент"
            lines.append(f"{speaker}: {cleaned}")
        transcript = "\n".join(lines)
        if len(transcript) > TRANSCRIPT_BUDGET_CHARS:
            # свежая часть разговора важнее: режем с начала по границе строки
            cut = transcript[-TRANSCRIPT_BUDGET_CHARS:]
            transcript = cut[cut.find("\n") + 1:] if "\n" in cut else cut
        return transcript, user_messages, user_chars

    async def _is_duplicate(self, fact: ExtractedFact) -> bool:
        """Уже есть похожий активный факт? Сравнение детерминированное:
        та же тема того же типа + сильное пересечение слов содержимого."""
        candidates = await self._store.search(
            self._owner, f"{fact.subject} {fact.content}", k=10
        )
        for existing in candidates:
            if _norm(existing.content) == _norm(fact.content):
                return True
            if existing.type != fact.type:
                continue
            if content_overlap(existing.content, fact.content) < DUPLICATE_OVERLAP:
                # у вытесняющих типов (решения) НЕпохожее содержимое по той же
                # теме — не дубль, а новое решение: пусть вытеснит старое
                continue
            if fact.type not in SUPERSEDING_TYPES:
                return True
            if _norm(existing.subject) == _norm(fact.subject):
                return True
        return False

    # ── журнал консолидаций ──────────────────────────────────────────────────

    async def _record(self, conversation_id: str, result: str, facts_added: int) -> None:
        await self._db.execute(
            "INSERT INTO memory_consolidations"
            " (conversation_id, processed_at, result, attempts, facts_added)"
            " VALUES (?, ?, ?, 1, ?)"
            " ON CONFLICT(conversation_id) DO UPDATE SET"
            " processed_at=excluded.processed_at, result=excluded.result,"
            " facts_added=excluded.facts_added",
            (conversation_id, datetime.now(UTC).isoformat(), result, facts_added),
        )

    async def _record_failure(self, conversation_id: str, error: str) -> None:
        log.warning(
            "memory_consolidation_failed", conversation=conversation_id, error=error
        )
        await self._db.execute(
            "INSERT INTO memory_consolidations"
            " (conversation_id, processed_at, result, attempts, facts_added)"
            " VALUES (?, ?, 'failed', 1, 0)"
            " ON CONFLICT(conversation_id) DO UPDATE SET"
            " processed_at=excluded.processed_at, result='failed',"
            " attempts=memory_consolidations.attempts+1",
            (conversation_id, datetime.now(UTC).isoformat()),
        )

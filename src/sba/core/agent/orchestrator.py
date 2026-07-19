"""Agent Orchestrator v2: agent loop с инструментами.

Цикл (docs/02-architecture.md §2.3): контекст → LLM (+tools) → если модель
позвала инструменты, исполнить и вернуть результаты в контекст → повторять
до текстового ответа или лимита итераций. destructive-вызовы прерывают цикл
вопросом пользователю; «да» следующим сообщением продолжает исполнение.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

from sba.core import execctx
from sba.core.agent.context import build_messages
from sba.core.agent.language import strip_cjk
from sba.core.history import HistoryReader
from sba.core.tools.registry import ConfirmationRequired, ToolRegistry
from sba.core.types import IncomingMessage, MemoryPort, Reply, Session
from sba.infra.audit import AuditLog
from sba.infra.config import AgentConfig
from sba.llm.gateway import ChatMessage, LLMError, LLMGateway, ToolCall
from sba.llm.toolcalling import parse_tool_call_text

log = structlog.get_logger(__name__)

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

CONFIRM_WORDS = {"да", "yes", "ок", "ok", "окей", "подтверждаю", "давай", "+"}
CANCEL_WORDS = {"нет", "no", "отмена", "отменить", "cancel", "стоп"}
PENDING_TTL_SECONDS = 300.0

# Маленькие модели пропускают вызов инструмента и отвечают «по памяти»,
# особенно если в истории уже есть их прошлый (возможно выдуманный) ответ.
# Подсказка вплотную к вопросу действует на них сильнее системного промпта.
FILE_TOPIC_MARKERS = (
    "файл", "папк", "найди", "найти", "поищи", "прочит", "покаж", "удали",
    "downloads", "documents", "загрузк", "документ", ".log", ".txt", ".md",
    ".pdf", ".docx", ".xlsx", "лог", "заметк", "конспект",
    # вопросы о наличии информации — это поиск по содержимому (search_documents)
    "инфа", "инфо", "информаци", "что говорится", "что написано", "что сказано",
)
FILE_NUDGE = (
    "Вопрос пользователя касается файлов или документов. ОБЯЗАТЕЛЬНО сначала "
    "вызови подходящий инструмент: search_documents — если вопрос о СОДЕРЖИМОМ "
    "документов или заметок; find_files или list_files — если нужен поиск/список "
    "файлов по именам; read_document — прочитать конкретный файл. Отвечай только "
    "по результату инструмента, с указанием файла-источника. Не отвечай по памяти. "
    "Не доверяй прошлым ответам из истории диалога — они могли быть ошибочными, "
    "проверь инструментом заново."
)
MEMORY_TOPIC_MARKERS = (
    "запомни", "запомн", "забудь", "забыть", "помнишь", "что ты знаешь",
    "кто такой", "кто такая", "мои предпочтения",
)
MEMORY_NUDGE = (
    "Сообщение касается памяти. ОБЯЗАТЕЛЬНО используй инструменты: "
    "remember_fact — чтобы запомнить, recall_memory — чтобы вспомнить, "
    "forget_memory — чтобы забыть. Не отвечай, что не умеешь запоминать, "
    "и не утверждай, что запомнил, без успешного вызова инструмента."
)
TASK_TOPIC_MARKERS = (
    "задач", "туду", "todo", "дедлайн", "отметь", "выполнен", "сделано",
    "по проекту", "напомни", "не забыть", "запланируй",
)
TASK_NUDGE = (
    "Сообщение касается задач. ОБЯЗАТЕЛЬНО используй инструменты задач: "
    "create_task — создать (срок передай словами в when), search_tasks — найти "
    "или показать список (в т.ч. «что у меня по проекту»), complete_task — "
    "отметить сделанной, update_task — изменить или отменить. Не сообщай, что "
    "создал или закрыл задачу, без успешного вызова инструмента. Отправлять "
    "напоминания сам ты ПОКА НЕ умеешь: на «напомни…» создай задачу со сроком "
    "и честно скажи, что уведомление в срок пока не придёт."
)
TOPIC_NUDGES: tuple[tuple[tuple[str, ...], str], ...] = (
    (FILE_TOPIC_MARKERS, FILE_NUDGE),
    (MEMORY_TOPIC_MARKERS, MEMORY_NUDGE),
    (TASK_TOPIC_MARKERS, TASK_NUDGE),
)

# Ollama игнорирует tool_choice=required (проверено вживую: модель отвечает
# текстом «из головы» на прямой вопрос о файлах). Принуждение выполняем сами:
# отказ от вызова → один строгий повтор → честный отказ вместо выдумки.
FORCE_RETRY_NUDGE = (
    "Ты ответил текстом, не вызвав инструмент, — так нельзя. Данные без "
    "инструмента считаются выдуманными. Сейчас же вызови подходящий инструмент."
)
FORCED_REFUSAL = (
    "⚠️ Модель дважды попыталась ответить без проверки инструментом — такой "
    "ответ может быть выдуман, поэтому я его не показываю. Переформулируйте "
    "вопрос (например: «найди в документах …») или начните новый разговор: /new."
)
# Растяжка на фабрикацию вне принудительных тем: в ответе упомянуты пути или
# файлы, хотя за весь ход не было ни одного реального вызова инструмента
PATH_MENTION_RE = re.compile(r"[A-Za-z]:\\|\.(docx|xlsx|pdf|txt|md|log)\b")
UNVERIFIED_PATH_WARNING = (
    "\n⚠️ В этом ответе инструменты не вызывались — упомянутые файлы могут "
    "быть выдуманы. Проверить реальные действия: /audit."
)


@dataclass
class PendingAction:
    """destructive-вызов, ожидающий подтверждения пользователя."""

    call: ToolCall
    messages: list[ChatMessage]  # контекст на момент прерывания (с tool_calls)
    created_at: float = field(default_factory=time.monotonic)

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.created_at > PENDING_TTL_SECONDS


def _normalize_answer(text: str) -> str:
    return text.strip().lower().rstrip("!.,)")


class AgentOrchestrator:
    def __init__(
        self,
        gateway: LLMGateway,
        history: HistoryReader,
        registry: ToolRegistry,
        audit: AuditLog,
        config: AgentConfig,
        timezone: str,
        memory: MemoryPort | None = None,
        extra_commands: dict[str, tuple[str, Callable[[], Awaitable[str]]]] | None = None,
    ) -> None:
        self._gateway = gateway
        self._history = history
        self._registry = registry
        self._audit = audit
        self._config = config
        self._memory = memory
        self._tz = ZoneInfo(timezone)
        self._template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self._pending: dict[tuple[str, str], PendingAction] = {}
        # сервис-команды модулей: (описание, обработчик); инъекция из app.py,
        # чтобы ядро не знало о модулях (границы docs/03)
        self._extra_commands = extra_commands or {}

    async def process(self, msg: IncomingMessage, session: Session) -> Reply:
        # привязка инструментов к источнику (задача ↔ сообщение, Sprint 5)
        execctx.current_message_id.set(msg.id)
        service_reply = await self._service_command(msg.text)
        if service_reply is not None:
            return service_reply

        key = (msg.user_id, msg.channel)
        pending = self._pending.pop(key, None)
        if pending is not None and not pending.expired:
            answer = _normalize_answer(msg.text)
            if answer in CONFIRM_WORDS:
                return self._confirmed_stream(pending, key)
            if answer in CANCEL_WORDS:
                log.info("destructive_cancelled", tool=pending.call.name)
                return "🚫 Действие отменено."
            # другое сообщение = молчаливая отмена, обрабатываем как обычно
            log.info("destructive_dropped", tool=pending.call.name)

        lowered = msg.text.lower()
        nudges = [
            text
            for markers, text in TOPIC_NUDGES
            if any(marker in lowered for marker in markers)
        ]
        messages = await self._build_context(msg, session, nudges=nudges)
        return self._agent_stream(messages, key, force_first_tool=bool(nudges))

    # ── служебные команды (мимо LLM, детерминированно) ───────────────────────

    async def _service_command(self, text: str) -> str | None:
        command = text.strip().lower()
        if command == "/tools":
            specs = self._registry.available()
            if not specs:
                return "Инструменты не зарегистрированы."
            return "Зарегистрированные инструменты:\n" + "\n".join(
                f"• {s.name} [{s.risk}] — {s.description}" for s in specs
            )
        if command == "/audit":
            rows = await self._audit.recent(12)
            if not rows:
                return (
                    "Журнал действий пуст: ни одного вызова инструмента ещё не было. "
                    "Если ассистент при этом рассказывал про файлы — он их выдумал."
                )
            lines = [
                f"{ts[11:19]} | {kind} | {name} | {detail[:90]}"
                for ts, kind, name, detail in rows
            ]
            return "Последние действия (новые сверху):\n" + "\n".join(lines)
        extra = self._extra_commands.get(command)
        if extra is not None:
            return await extra[1]()
        if command.startswith("/"):
            known = [
                "/new — новый разговор",
                "/tools — список инструментов",
                "/audit — журнал действий",
                *(f"{name} — {descr}" for name, (descr, _) in self._extra_commands.items()),
            ]
            return f"Неизвестная команда {command}. Доступны: " + ", ".join(known) + "."
        return None

    # ── построение контекста ─────────────────────────────────────────────────

    async def _build_context(
        self, msg: IncomingMessage, session: Session, nudges: list[str]
    ) -> list[ChatMessage]:
        # история уже содержит текущее сообщение (Router сохраняет его до обработки)
        entries = await self._history.recent(session.id, self._config.history_max_messages)
        now = datetime.now(self._tz)
        system = self._template.format(
            now=now.strftime("%Y-%m-%d %H:%M, %A"), timezone=self._tz.key
        )
        if self._memory is not None:
            preferences = await self._memory.preferences_text()
            if preferences:
                system += (
                    "\n\nПРЕДПОЧТЕНИЯ ВЛАДЕЛЬЦА (всегда следуй им):\n" + preferences
                )
        messages = build_messages(system, entries, self._config.history_budget_chars)
        if self._memory is not None:
            facts = await self._memory.relevant_facts_text(msg.text)
            if facts:
                messages.append(
                    ChatMessage(
                        role="system",
                        content="ФАКТЫ ИЗ ДОЛГОВРЕМЕННОЙ ПАМЯТИ (проверенные, "
                        "используй при ответе):\n" + facts,
                    )
                )
        for nudge in nudges:
            messages.append(ChatMessage(role="system", content=nudge))
        return messages

    # ── agent loop ───────────────────────────────────────────────────────────

    async def _agent_stream(
        self,
        messages: list[ChatMessage],
        key: tuple[str, str],
        force_first_tool: bool = False,
        tool_already_executed: bool = False,
    ) -> AsyncIterator[str]:
        tools = self._registry.openai_schemas()
        shown_any = False
        executed: dict[tuple[str, str], str] = {}  # дедуп повторных одинаковых вызовов
        # принуждение к инструменту: держится, пока не случится реальный вызов
        force_pending = force_first_tool
        forced_retry_used = False
        any_tool_executed = tool_already_executed
        try:
            for iteration in range(self._config.max_tool_iterations):
                tool_choice = "required" if force_pending else None
                # на принудительном шаге текст буферизуем: если модель вместо
                # вызова начнёт сочинять ответ, пользователь его не увидит
                buffering = force_pending
                raw_text: list[str] = []
                tool_calls: list[ToolCall] | None = None
                async for event in self._gateway.stream(
                    "chat", messages, tools=tools, tool_choice=tool_choice
                ):
                    if event.text:
                        raw_text.append(event.text)
                        if not buffering:
                            cleaned = strip_cjk(event.text)  # языковой барьер
                            if cleaned:
                                shown_any = True
                                yield cleaned
                    if event.tool_calls:
                        tool_calls = event.tool_calls

                joined = "".join(raw_text).strip()
                if not tool_calls:
                    # qwen иногда пишет вызов инструмента JSON-текстом в ответ —
                    # спасаем его как настоящий вызов, а не показываем мусор
                    salvaged = parse_tool_call_text(joined) if joined else None
                    if salvaged is not None:
                        log.info("text_tool_call_salvaged", tool=salvaged[0].name)
                        tool_calls = salvaged
                    elif force_pending:
                        # рантайм проигнорировал tool_choice=required (Ollama так
                        # делает), модель ответила «из головы» — текст не показываем
                        if not forced_retry_used:
                            forced_retry_used = True
                            log.warning("forced_tool_ignored_retrying")
                            messages.append(
                                ChatMessage(role="system", content=FORCE_RETRY_NUDGE)
                            )
                            continue
                        log.error("forced_tool_refused", discarded=joined[:200])
                        yield FORCED_REFUSAL
                        return
                    else:
                        if not shown_any:
                            yield (
                                "(модель ответила не на русском — переформулируйте "
                                "вопрос, пожалуйста)"
                                if raw_text
                                else "(модель вернула пустой ответ)"
                            )
                        elif not any_tool_executed and PATH_MENTION_RE.search(joined):
                            # ответ называет файлы, хотя инструменты не вызывались
                            log.warning("path_mention_without_tools")
                            yield UNVERIFIED_PATH_WARNING
                        return
                force_pending = False  # реальный вызов состоялся

                log.info(
                    "agent_tool_round",
                    iteration=iteration,
                    tools=[c.name for c in tool_calls],
                )
                messages.append(
                    ChatMessage(
                        role="assistant", content="".join(raw_text), tool_calls=tool_calls
                    )
                )
                for call in tool_calls:
                    signature = (
                        call.name,
                        json.dumps(call.arguments, sort_keys=True, ensure_ascii=False),
                    )
                    if signature in executed:
                        # модель зациклилась на одном вызове: не жжём минуты CPU,
                        # а прямо говорим ей сформулировать ответ
                        log.info("duplicate_tool_call_skipped", tool=call.name)
                        messages.append(
                            ChatMessage(
                                role="tool",
                                tool_call_id=call.id,
                                content="(повторный вызов с теми же аргументами; "
                                "результат не изменился — он уже есть выше. "
                                "Сформулируй ответ пользователю по этим данным.)",
                            )
                        )
                        continue
                    # видимый маркер реального вызова — защита доверия: ответ
                    # про файлы без строки 🔧 означает, что модель сочиняет
                    args_preview = json.dumps(call.arguments, ensure_ascii=False)
                    if len(args_preview) > 120:
                        args_preview = args_preview[:120] + "…"
                    shown_any = True
                    yield f"🔧 {call.name}({args_preview})\n"
                    try:
                        result = await self._registry.execute(call)
                    except ConfirmationRequired as need:
                        self._pending[key] = PendingAction(call=need.call, messages=messages)
                        args = json.dumps(call.arguments, ensure_ascii=False)
                        yield (
                            f"{chr(10) if shown_any else ''}🛑 Действие требует "
                            f"подтверждения:\n{need.spec.description}\n"
                            f"Инструмент: {call.name}, аргументы: {args}\n\n"
                            "Ответьте «да» для выполнения или «нет» для отмены."
                        )
                        return
                    executed[signature] = result.text
                    any_tool_executed = True
                    messages.append(
                        ChatMessage(role="tool", tool_call_id=call.id, content=result.text)
                    )

            yield (
                f"{chr(10) if shown_any else ''}⚠️ Достиг лимита шагов "
                f"({self._config.max_tool_iterations}) и не довёл дело до конца. "
                "Попробуйте разбить задачу на части."
            )
        except LLMError as exc:
            log.error("llm_failed", error=str(exc))
            yield (
                f"{chr(10) if shown_any else ''}⚠️ Не получилось обратиться к модели: {exc}\n"
                "Если это первый вопрос после простоя или запуска — модель "
                "загружалась в память: просто повторите сообщение. Иначе проверьте, "
                "что Ollama запущена (ollama ps)."
            )

    # ── продолжение после подтверждения ──────────────────────────────────────

    async def _confirmed_stream(
        self, pending: PendingAction, key: tuple[str, str]
    ) -> AsyncIterator[str]:
        result = await self._registry.execute(pending.call, confirmed=True)
        messages = [
            *pending.messages,
            ChatMessage(role="tool", tool_call_id=pending.call.id, content=result.text),
        ]
        async for chunk in self._agent_stream(messages, key, tool_already_executed=True):
            yield chunk

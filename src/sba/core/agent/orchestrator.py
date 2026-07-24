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
# 15 минут: человек в Telegram отвечает «да» не сразу (отлучился, отвлёкся);
# 5 минут оказалось мало (живая проверка Sprint 8). Позже этого срок истекает,
# и «да» получает честный отказ, а не выполнение забытого действия
PENDING_TTL_SECONDS = 900.0

# Маленькие модели пропускают вызов инструмента и отвечают «по памяти»,
# особенно если в истории уже есть их прошлый (возможно выдуманный) ответ.
# Подсказка вплотную к вопросу действует на них сильнее системного промпта.
FILE_TOPIC_MARKERS = (
    "файл", "папк", "найди", "найти", "поищи", "прочит", "покаж", "удали",
    "downloads", "documents", "загрузк", "документ", ".log", ".txt", ".md",
    ".pdf", ".docx", ".xlsx", "лог", "заметк", "конспект",
    # вопросы о наличии информации — это поиск по содержимому (search_documents)
    "инфа", "инфо", "информаци", "что говорится", "что написано", "что сказано",
    # файловые операции (Sprint 8): порядок, перенос, дубликаты, архив, откат
    "перемест", "скопир", "переимен", "разбери", "разложи", "дубл", "архив",
    "порядок", "операци",
)
FILE_NUDGE = (
    "Вопрос пользователя касается файлов или документов. ОБЯЗАТЕЛЬНО сначала "
    "вызови подходящий инструмент: search_documents — если вопрос о СОДЕРЖИМОМ "
    "документов или заметок; find_files или list_files — если нужен поиск/список "
    "файлов по именам; read_document — прочитать конкретный файл. Для наведения "
    "порядка: move_file / copy_file / rename_file — переместить, скопировать, "
    "переименовать; archive_files — убрать файлы папки в архив; find_duplicates / "
    "find_old_versions — найти дубликаты и старые версии; undo_file_operation — "
    "откатить последнюю операцию. Отвечай только по результату инструмента, с "
    "указанием файла-источника. Не отвечай по памяти. Не доверяй прошлым ответам "
    "из истории диалога — они могли быть ошибочными, проверь инструментом заново. "
    "Если инструмент ответил «СУХОЙ ПРОГОН» — операция НЕ выполнена: передай "
    "владельцу план и скажи, что файлы не тронуты."
)
MEMORY_TOPIC_MARKERS = (
    "запомни", "запомн", "забудь", "забыть", "помнишь", "что ты знаешь",
    "кто такой", "кто такая", "мои предпочтения",
    # вопросы о прошлых разговорах — recall по эпизодам (Sprint 7)
    "вспомн", "о чём мы", "о чем мы", "обсужда", "говорили",
    # разговорные формулировки вопроса к памяти: «про X знаешь что-то?»,
    # «что знаешь о…», «знаешь про…» — иначе модель отвечает «из головы»
    "знаешь про", "знаешь о", "знаешь что", "что знаешь", "знаешь ли",
)
MEMORY_NUDGE = (
    "Сообщение касается памяти. ОБЯЗАТЕЛЬНО используй инструменты: "
    "remember_fact — чтобы запомнить, recall_memory — чтобы вспомнить, "
    "forget_memory — чтобы забыть. Не отвечай, что не умеешь запоминать, "
    "и не утверждай, что запомнил, без успешного вызова инструмента."
)
TASK_TOPIC_MARKERS = (
    "задач", "туду", "todo", "дедлайн", "отметь", "выполнен", "сделано",
    "по проекту", "напом", "не забыть", "не забудь", "не забывай",
    "запланируй", "ежеднев", "еженедель", "ежемесяч", "кажд", "по утрам",
)
TASK_NUDGE = (
    "Сообщение касается задач или напоминаний. И то и другое ты УМЕЕШЬ — не "
    "говори, что это недоступно. ОБЯЗАТЕЛЬНО используй инструменты: "
    "create_reminder — «напомни/напоминай…» (само придёт в срок; повторение "
    "словами в when; «пока не сделаю» — repeat_until_done=true), create_task — "
    "«поставь задачу / добавь в дела» (срок словами в when), search_tasks и "
    "list_reminders — найти или показать список, complete_task — отметить "
    "сделанной, update_task — изменить или отменить задачу, snooze_reminder / "
    "cancel_reminder — перенести или отменить напоминание. Не сообщай, что "
    "создал, перенёс или закрыл, без успешного вызова инструмента. Различай: "
    "ВОПРОС о том, что уже есть («напомни/подскажи, когда/во сколько…», «что у "
    "меня…») — это поиск: вызови search_tasks или list_reminders и ответь по "
    "найденному, НЕ создавай новое. ПРОСЬБА о будущем напоминании («напомни "
    "завтра…», «напоминай каждое утро…») — вызови create_reminder."
)
# Каждая тема несёт и подсказку, и набор модулей-инструментов: при
# topic_scoped_tools модели уходят только релевантные теме инструменты
# (короче промпт → быстрее ответ на CPU, меньше путаницы у слабой модели).
TOPIC_RULES: tuple[tuple[tuple[str, ...], str, frozenset[str]], ...] = (
    (FILE_TOPIC_MARKERS, FILE_NUDGE, frozenset({"files", "rag"})),
    (MEMORY_TOPIC_MARKERS, MEMORY_NUDGE, frozenset({"memory"})),
    (TASK_TOPIC_MARKERS, TASK_NUDGE, frozenset({"tasks", "reminders"})),
)
# get_current_time дёшев и полезен для расчёта дат — доступен всегда
ALWAYS_MODULES = frozenset({"basic"})

# Ollama игнорирует tool_choice=required (проверено вживую: модель отвечает
# текстом «из головы» на прямой вопрос о файлах). Принуждение выполняем сами:
# отказ от вызова → один строгий повтор → честный отказ вместо выдумки.
FORCE_RETRY_NUDGE = (
    "Ты ответил текстом, не вызвав инструмент, — так нельзя. Данные без "
    "инструмента считаются выдуманными. Сейчас же вызови подходящий инструмент."
)
FORCED_REFUSAL = (
    "⚠️ Модель дважды попыталась ответить без проверки инструментом — такой "
    "ответ может быть выдуман, поэтому я его не показываю. Повторите запрос чуть "
    "конкретнее (что именно найти, создать или отметить) или начните новый "
    "разговор: /new."
)
# 7B после первого вызова иногда выдаёт СЛЕДУЮЩИЕ вызовы не штатно, а JSON-текстом
# (в т.ч. массивом [{...}, {...}]) — это утекало сырым в чат (живая проверка
# Sprint 8). Такой текст придерживаем и не показываем: просим ответить по-человечески.
JSON_TEXT_RETRY_NUDGE = (
    "Ты вывел вызов инструмента служебным JSON-текстом — пользователь не должен "
    "видеть служебный формат. Если нужен инструмент — вызови его штатно, не "
    "текстом. Если данных уже достаточно — ответь пользователю обычным текстом "
    "по-русски по полученным результатам."
)
JSON_TEXT_REFUSAL = (
    "⚠️ Модель выдавала ответ служебным форматом вместо текста, поэтому я его не "
    "показываю. Повторите запрос или начните новый разговор: /new."
)
# Растяжка на фабрикацию вне принудительных тем: в ответе упомянуты пути или
# файлы, хотя за весь ход не было ни одного реального вызова инструмента
PATH_MENTION_RE = re.compile(r"[A-Za-z]:\\|\.(docx|xlsx|pdf|txt|md|log)\b")
UNVERIFIED_PATH_WARNING = (
    "\n⚠️ В этом ответе инструменты не вызывались — упомянутые файлы могут "
    "быть выдуманы. Проверить реальные действия: /audit."
)
# Та же защита для задач и напоминаний: модель без вызова инструмента заявляет
# «создана/закрыта задача №…» и выдаёт пример id из промпта за настоящий
# (подтверждено: /tasks пуст)
TASK_CLAIM_RE = re.compile(
    r"(созда|закры|удал|отмен|отмеч|выполн|перенёс|перенес|обнов)\w*\s+"
    r"(задач|напоминани)|(задач|напоминани)\w*\s*"
    r"(созда|закры|удал|отмен|отмеч|выполн|№|#)|буду напоминать|напомню в?\s*\d",
    re.IGNORECASE,
)
UNVERIFIED_TASK_WARNING = (
    "\n⚠️ Ни один инструмент задач или напоминаний не вызывался — на самом деле "
    "ничего НЕ создано и не изменено, а номер мог быть выдуман. Проверьте "
    "списки: /tasks и /reminders."
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
        extra_always_modules: set[str] | None = None,
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
        # модули, чьи инструменты доступны модели всегда, даже при
        # topic_scoped_tools (MCP-серверы, Sprint 9: у их тем нет наших маркеров).
        # Держим ссылку как есть (а не `or set()`): MCP подключается фоново, и
        # пустое на момент создания множество хаб пополнит позже — по этой же
        # ссылке. Новый set() порвал бы связь и MCP-тулы не попадали бы в scope
        self._extra_always = extra_always_modules if extra_always_modules is not None else set()

    async def process(self, msg: IncomingMessage, session: Session) -> Reply:
        # привязка инструментов к источнику (задача ↔ сообщение, Sprint 5)
        execctx.current_message_id.set(msg.id)
        service_reply = await self._service_command(msg.text)
        if service_reply is not None:
            return service_reply

        key = (msg.user_id, msg.channel)
        pending = self._pending.pop(key, None)
        if pending is not None:
            answer = _normalize_answer(msg.text)
            if pending.expired:
                # «да»/«нет» спустя время: подтверждать нечего. Но НЕ отдаём это
                # модели — иначе слабая модель сочинит «готово» (живая проверка
                # Sprint 8). Честно говорим, что срок истёк и ничего не сделано.
                if answer in CONFIRM_WORDS or answer in CANCEL_WORDS:
                    log.info("destructive_expired", tool=pending.call.name)
                    return (
                        "⌛ Срок подтверждения истёк — действие НЕ выполнено. "
                        "Повторите запрос, если он ещё нужен."
                    )
                log.info("destructive_expired_dropped", tool=pending.call.name)
            elif answer in CONFIRM_WORDS:
                return self._confirmed_stream(pending, key)
            elif answer in CANCEL_WORDS:
                log.info("destructive_cancelled", tool=pending.call.name)
                return "🚫 Действие отменено."
            else:
                # другое сообщение = молчаливая отмена, обрабатываем как обычно
                log.info("destructive_dropped", tool=pending.call.name)

        lowered = msg.text.lower()
        nudges: list[str] = []
        allowed_modules: set[str] = set(ALWAYS_MODULES) | self._extra_always
        for markers, nudge, modules in TOPIC_RULES:
            if any(marker in lowered for marker in markers):
                nudges.append(nudge)
                allowed_modules |= modules
        messages = await self._build_context(msg, session, nudges=nudges)
        # scope=None → все инструменты (по умолчанию); при topic_scoped_tools
        # шлём только релевантные теме (на общую реплику без темы — только basic)
        scope = allowed_modules if self._config.topic_scoped_tools else None
        return self._agent_stream(
            messages, key, allowed_modules=scope, force_first_tool=bool(nudges)
        )

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
        allowed_modules: set[str] | None = None,
    ) -> AsyncIterator[str]:
        tools = self._registry.openai_schemas(allowed_modules)
        shown_any = False
        executed: dict[tuple[str, str], str] = {}  # дедуп повторных одинаковых вызовов
        # принуждение к инструменту: держится, пока не случится реальный вызов
        force_pending = force_first_tool
        forced_retry_used = False
        json_retry_used = False
        any_tool_executed = False
        try:
            for iteration in range(self._config.max_tool_iterations):
                tool_choice = "required" if force_pending else None
                # на принудительном шаге текст буферизуем целиком: если модель
                # вместо вызова начнёт сочинять ответ, пользователь его не увидит
                buffering = force_pending
                raw_text: list[str] = []
                tool_calls: list[ToolCall] | None = None
                # «нюхаем» начало свободного ответа: обычный текст начинается с
                # буквы — стримим сразу (потоковость сохранена); текст, начатый
                # с { или [ — вероятный вызов инструмента, выданный JSON-ом (7B
                # так делает после первого вызова) — придерживаем и НЕ показываем
                # сырой служебный формат. live: None=нюхаем, True=стримим, False=держим.
                live: bool | None = False if buffering else None
                sniff = ""
                async for event in self._gateway.stream(
                    "chat", messages, tools=tools, tool_choice=tool_choice
                ):
                    if event.text:
                        raw_text.append(event.text)
                        if live is None:
                            sniff += event.text
                            head = sniff.lstrip()
                            if head:
                                if head[0] in "{[":
                                    live = False  # похоже на JSON-вызов — придержать
                                else:
                                    live = True
                                    cleaned = strip_cjk(sniff)  # языковой барьер
                                    if cleaned:
                                        shown_any = True
                                        yield cleaned
                                    sniff = ""
                        elif live:
                            cleaned = strip_cjk(event.text)  # языковой барьер
                            if cleaned:
                                shown_any = True
                                yield cleaned
                    if event.tool_calls:
                        tool_calls = event.tool_calls

                joined = "".join(raw_text).strip()
                if not tool_calls:
                    # qwen иногда пишет вызов инструмента JSON-текстом в ответ —
                    # спасаем одиночный вызов как настоящий, а не показываем мусор
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
                    elif live is False:
                        # придержали JSON-подобный текст, но одиночным валидным
                        # вызовом он не оказался (массив вызовов или мусор — 7B
                        # имитирует служебный формат). Сырой JSON не показываем:
                        # один раз просим ответить по-человечески, затем честно
                        # сообщаем о срыве вместо вывода служебных строк.
                        if not json_retry_used:
                            json_retry_used = True
                            log.warning("text_tool_json_held_retrying")
                            messages.append(
                                ChatMessage(role="system", content=JSON_TEXT_RETRY_NUDGE)
                            )
                            continue
                        log.error("text_tool_json_refused", discarded=joined[:200])
                        yield JSON_TEXT_REFUSAL
                        return
                    else:
                        if not shown_any:
                            yield (
                                "(модель ответила не на русском — переформулируйте "
                                "вопрос, пожалуйста)"
                                if raw_text
                                else "(модель вернула пустой ответ)"
                            )
                        elif not any_tool_executed and TASK_CLAIM_RE.search(joined):
                            # ответ заявляет действие над задачей без вызова инструмента
                            log.warning("task_claim_without_tools")
                            yield UNVERIFIED_TASK_WARNING
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
        # Опасное действие пользователь одобрил явно — исполняем и СРАЗУ показываем
        # результат, не пуская модель в новый круг agent-loop. Живая проверка
        # Sprint 8: слабая 3B/7B после подтверждения срывалась на выдуманные вызовы
        # с пустыми аргументами и «воду»; крутить ещё проход LLM тут незачем —
        # результат инструмента уже готов к показу пользователю.
        result = await self._registry.execute(pending.call, confirmed=True)
        yield ("⚠️ " if result.error else "✅ ") + result.text

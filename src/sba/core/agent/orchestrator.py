"""Agent Orchestrator v2: agent loop с инструментами.

Цикл (docs/02-architecture.md §2.3): контекст → LLM (+tools) → если модель
позвала инструменты, исполнить и вернуть результаты в контекст → повторять
до текстового ответа или лимита итераций. destructive-вызовы прерывают цикл
вопросом пользователю; «да» следующим сообщением продолжает исполнение.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

from sba.core.agent.context import build_messages
from sba.core.agent.language import strip_cjk
from sba.core.history import HistoryReader
from sba.core.tools.registry import ConfirmationRequired, ToolRegistry
from sba.core.types import IncomingMessage, Reply, Session
from sba.infra.audit import AuditLog
from sba.infra.config import AgentConfig
from sba.llm.gateway import ChatMessage, LLMError, LLMGateway, ToolCall

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
    ".pdf", ".docx", ".xlsx", "лог",
)
TOOL_NUDGE = (
    "Вопрос пользователя касается файлов. ОБЯЗАТЕЛЬНО сначала вызови подходящий "
    "инструмент (find_files, list_files или read_document) и отвечай только по "
    "его результату. Не отвечай по памяти. Не доверяй прошлым ответам из "
    "истории диалога — они могли быть ошибочными, проверь инструментом заново."
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
    ) -> None:
        self._gateway = gateway
        self._history = history
        self._registry = registry
        self._audit = audit
        self._config = config
        self._tz = ZoneInfo(timezone)
        self._template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        self._pending: dict[tuple[str, str], PendingAction] = {}

    async def process(self, msg: IncomingMessage, session: Session) -> Reply:
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
        file_topic = any(marker in lowered for marker in FILE_TOPIC_MARKERS)
        messages = await self._build_context(msg, session, nudge=file_topic)
        return self._agent_stream(messages, key, force_first_tool=file_topic)

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
        if command.startswith("/"):
            return (
                f"Неизвестная команда {command}. Доступны: /new — новый разговор, "
                "/tools — список инструментов, /audit — журнал действий."
            )
        return None

    # ── построение контекста ─────────────────────────────────────────────────

    async def _build_context(
        self, msg: IncomingMessage, session: Session, nudge: bool
    ) -> list[ChatMessage]:
        # история уже содержит текущее сообщение (Router сохраняет его до обработки)
        entries = await self._history.recent(session.id, self._config.history_max_messages)
        now = datetime.now(self._tz)
        system = self._template.format(
            now=now.strftime("%Y-%m-%d %H:%M, %A"), timezone=self._tz.key
        )
        messages = build_messages(system, entries, self._config.history_budget_chars)
        if nudge:
            messages.append(ChatMessage(role="system", content=TOOL_NUDGE))
        return messages

    # ── agent loop ───────────────────────────────────────────────────────────

    async def _agent_stream(
        self,
        messages: list[ChatMessage],
        key: tuple[str, str],
        force_first_tool: bool = False,
    ) -> AsyncIterator[str]:
        tools = self._registry.openai_schemas()
        shown_any = False
        executed: dict[tuple[str, str], str] = {}  # дедуп повторных одинаковых вызовов
        try:
            for iteration in range(self._config.max_tool_iterations):
                # required только на первом шаге: дальше модель должна уметь
                # завершить ответ текстом (если рантайм вообще поддерживает это поле)
                tool_choice = "required" if force_first_tool and iteration == 0 else None
                raw_text: list[str] = []
                tool_calls: list[ToolCall] | None = None
                async for event in self._gateway.stream(
                    "chat", messages, tools=tools, tool_choice=tool_choice
                ):
                    if event.text:
                        raw_text.append(event.text)
                        cleaned = strip_cjk(event.text)  # языковой барьер
                        if cleaned:
                            shown_any = True
                            yield cleaned
                    if event.tool_calls:
                        tool_calls = event.tool_calls

                if not tool_calls:
                    if not shown_any:
                        yield (
                            "(модель ответила не на русском — переформулируйте "
                            "вопрос, пожалуйста)"
                            if raw_text
                            else "(модель вернула пустой ответ)"
                        )
                    return

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
                "Проверьте, что Ollama запущена (ollama ps)."
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
        async for chunk in self._agent_stream(messages, key):
            yield chunk

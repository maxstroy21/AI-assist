from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from sba.core.agent.orchestrator import AgentOrchestrator
from sba.core.history import HistoryEntry
from sba.core.tools.registry import ToolRegistry
from sba.core.tools.spec import RiskLevel, ToolSpec
from sba.core.types import IncomingMessage, Session, utcnow
from sba.infra.audit import AuditLog
from sba.infra.config import AgentConfig
from sba.infra.db import Database
from sba.llm.gateway import ChatMessage, LLMError, Role, StreamEvent, ToolCall, ToolSchema
from sba.llm.providers.fake import FakeLLM


class StubHistory:
    def __init__(self, entries: list[HistoryEntry]) -> None:
        self.entries = entries

    async def recent(self, conversation_id: str, limit: int) -> list[HistoryEntry]:
        return self.entries[-limit:]


class FailingLLM:
    def __init__(self, error: str = "нет соединения", local: bool = True) -> None:
        self._error = error
        self._local = local

    def chat_runtime_is_local(self) -> bool:
        return self._local

    async def chat(self, role: Role, messages: list[ChatMessage]):  # pragma: no cover
        raise LLMError(self._error)

    async def stream(
        self,
        role: Role,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        tool_choice: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        raise LLMError(self._error)
        yield StreamEvent()  # unreachable, делает функцию генератором


class ProbeArgs(BaseModel):
    value: str = "?"


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def probe_spec(executed: list[str], risk: RiskLevel = RiskLevel.READ) -> ToolSpec:
    async def handler(args: BaseModel) -> str:
        assert isinstance(args, ProbeArgs)
        executed.append(args.value)
        return f"результат:{args.value}"

    return ToolSpec(
        name="probe",
        description="тестовый инструмент",
        args_schema=ProbeArgs,
        risk=risk,
        module="test",
        handler=handler,
    )


class StubMemory:
    def __init__(self, preferences: str | None = None, facts: str | None = None) -> None:
        self._preferences = preferences
        self._facts = facts
        self.fact_queries: list[str] = []

    async def preferences_text(self) -> str | None:
        return self._preferences

    async def relevant_facts_text(self, query: str) -> str | None:
        self.fact_queries.append(query)
        return self._facts


def make_orchestrator(
    llm,
    db: Database,
    history: list[HistoryEntry] | None = None,
    tools: list[ToolSpec] = (),
    memory: StubMemory | None = None,
    **config,
) -> AgentOrchestrator:
    audit = AuditLog(db)
    registry = ToolRegistry(audit)
    for spec in tools:
        registry.register(spec)
    return AgentOrchestrator(
        gateway=llm,
        history=StubHistory(history or [HistoryEntry("user", "вопрос")]),
        registry=registry,
        audit=audit,
        config=AgentConfig(**config),
        timezone="Europe/Moscow",
        memory=memory,
    )


def make_session() -> Session:
    now = utcnow()
    return Session(id="s1", user_id="u1", channel="cli", started_at=now, last_activity_at=now)


async def collect(orchestrator: AgentOrchestrator, text: str = "вопрос") -> str:
    reply = await orchestrator.process(
        IncomingMessage(user_id="u1", channel="cli", text=text), make_session()
    )
    if isinstance(reply, str):
        return reply
    return "".join([d async for d in reply])


# ── базовый диалог (без инструментов в сценарии) ─────────────────────────────


async def test_plain_reply_streamed(db: Database) -> None:
    llm = FakeLLM(replies=["Привет! Чем помочь?"])
    assert await collect(make_orchestrator(llm, db)) == "Привет! Чем помочь?"


async def test_context_contains_system_and_history(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    history = [
        HistoryEntry("user", "меня зовут Макс"),
        HistoryEntry("assistant", "приятно познакомиться"),
        HistoryEntry("user", "как меня зовут?"),
    ]
    await collect(make_orchestrator(llm, db, history=history))

    role, messages = llm.calls[0]
    assert role == "chat"
    assert messages[0].role == "system"
    assert "Второй мозг" in messages[0].content
    assert [(m.role, m.content) for m in messages[1:]] == [
        ("user", "меня зовут Макс"),
        ("assistant", "приятно познакомиться"),
        ("user", "как меня зовут?"),
    ]


async def test_history_trimmed_to_budget_keeps_latest(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    history = [
        HistoryEntry("user", "старое " * 200),
        HistoryEntry("assistant", "среднее " * 200),
        HistoryEntry("user", "свежий вопрос"),
    ]
    await collect(make_orchestrator(llm, db, history=history, history_budget_chars=100))

    _, messages = llm.calls[0]
    assert len(messages) == 2  # system + только свежайшее
    assert messages[-1].content == "свежий вопрос"


async def test_llm_failure_local_hint_mentions_ollama(db: Database) -> None:
    text = await collect(make_orchestrator(FailingLLM(local=True), db))
    assert "⚠️" in text
    assert "Ollama" in text


async def test_llm_failure_cloud_429_hint_mentions_limit_not_ollama(db: Database) -> None:
    llm = FailingLLM(error="HTTP 429: rate limit", local=False)
    text = await collect(make_orchestrator(llm, db))
    assert "лимит" in text.lower()
    assert "Ollama" not in text  # облаку подсказка про Ollama не к месту


async def test_llm_failure_cloud_generic_hint_mentions_key(db: Database) -> None:
    llm = FailingLLM(error="HTTP 401: bad key", local=False)
    text = await collect(make_orchestrator(llm, db))
    assert "ключ" in text.lower()
    assert "Ollama" not in text


async def test_service_command_tools_bypasses_llm(db: Database) -> None:
    llm = FakeLLM()
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec([])]), "/tools")
    assert "probe" in text
    assert llm.calls == []  # модель не участвовала


async def test_service_command_audit_empty_and_after_call(db: Database) -> None:
    executed: list[str] = []
    llm = FakeLLM(
        replies=[[ToolCall(id="c1", name="probe", arguments={"value": "x"})], "готово"]
    )
    orchestrator = make_orchestrator(llm, db, tools=[probe_spec(executed)])

    empty = await collect(orchestrator, "/audit")
    assert "пуст" in empty

    await collect(orchestrator, "сделай probe")
    report = await collect(orchestrator, "/audit")
    assert "tool_call" in report and "probe" in report


async def test_unknown_slash_command_gets_help_without_llm(db: Database) -> None:
    llm = FakeLLM()
    text = await collect(make_orchestrator(llm, db), "/помощь")
    assert "Доступны" in text
    assert llm.calls == []


async def test_file_question_gets_tool_nudge(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    await collect(make_orchestrator(llm, db), "найди все файлы .log")
    _, messages = llm.calls[0]
    assert messages[-1].role == "system"
    assert "инструмент" in messages[-1].content


async def test_memory_question_colloquial_phrasing_gets_nudge(db: Database) -> None:
    # «про X знаешь что-то?» раньше не распознавалось — модель отвечала «из
    # головы», не заглянув в память (живой отчёт владельца, Sprint 7)
    llm = FakeLLM(replies=["ок"])
    await collect(make_orchestrator(llm, db), "про Кузнецова знаешь что-то?")
    _, messages = llm.calls[0]
    assert messages[-1].role == "system"
    assert "recall_memory" in messages[-1].content


async def test_smalltalk_gets_no_nudge(db: Database) -> None:
    llm = FakeLLM(replies=["привет!"])
    await collect(make_orchestrator(llm, db), "привет, как дела?")
    _, messages = llm.calls[0]
    assert messages[-1].role == "user"
    assert llm.seen_tool_choice == [None]


async def test_file_question_forces_tool_on_first_round_only(db: Database) -> None:
    llm = FakeLLM(
        replies=[[ToolCall(id="c1", name="probe", arguments={"value": "x"})], "готово"]
    )
    await collect(make_orchestrator(llm, db, tools=[probe_spec([])]), "найди файл отчёт")
    assert llm.seen_tool_choice == ["required", None]


async def test_preferences_injected_into_system_prompt(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    memory = StubMemory(preferences="- стиль ответов: отвечать кратко")
    await collect(make_orchestrator(llm, db, memory=memory))
    _, messages = llm.calls[0]
    assert "ПРЕДПОЧТЕНИЯ ВЛАДЕЛЬЦА" in messages[0].content
    assert "кратко" in messages[0].content


async def test_relevant_facts_injected_near_question(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    memory = StubMemory(facts="- [человек] Иван Петров: подрядчик")
    await collect(make_orchestrator(llm, db, memory=memory), "кто делает смету?")
    _, messages = llm.calls[0]
    assert any(
        m.role == "system" and "Иван Петров" in m.content for m in messages[1:]
    )  # факт добавлен отдельным system-сообщением рядом с вопросом
    assert memory.fact_queries == ["кто делает смету?"]


async def test_no_memory_no_injection(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    await collect(make_orchestrator(llm, db))
    _, messages = llm.calls[0]
    system_messages = [m for m in messages if m.role == "system"]
    assert len(system_messages) == 1  # только базовый промпт: ни фактов, ни предпочтений
    assert "ПРЕДПОЧТЕНИЯ ВЛАДЕЛЬЦА" not in system_messages[0].content


async def test_memory_question_forces_tool(db: Database) -> None:
    llm = FakeLLM(
        replies=[[ToolCall(id="c1", name="probe", arguments={"value": "x"})], "запомнил"]
    )
    await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])]),
        "запомни: у мамы день рождения 3 марта",
    )
    assert llm.seen_tool_choice[0] == "required"
    _, messages = llm.calls[0]
    assert any("remember_fact" in m.content for m in messages if m.role == "system")


# ── agent loop с инструментами ───────────────────────────────────────────────


async def test_tool_round_then_final_answer(db: Database) -> None:
    executed: list[str] = []
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "x"})],
            "Готово: probe вернул результат",
        ]
    )
    orchestrator = make_orchestrator(llm, db, tools=[probe_spec(executed)])

    text = await collect(orchestrator)
    assert "🔧 probe" in text          # видимый маркер реального вызова
    assert text.endswith("Готово: probe вернул результат")
    assert executed == ["x"]
    assert llm.seen_tools[0]  # схемы инструментов переданы модели

    # во втором вызове модель видит результат инструмента
    _, second_messages = llm.calls[1]
    tool_msgs = [m for m in second_messages if m.role == "tool"]
    assert tool_msgs and tool_msgs[0].content == "результат:x"
    assert tool_msgs[0].tool_call_id == "c1"


def _module_tool(name: str, module: str) -> ToolSpec:
    async def handler(args: BaseModel) -> str:
        return f"{name}:ok"

    return ToolSpec(
        name=name, description=f"{name} инструмент", args_schema=ProbeArgs,
        risk=RiskLevel.READ, module=module, handler=handler,
    )


async def test_topic_scoped_tools_sends_only_relevant_modules(db: Database) -> None:
    """topic_scoped_tools: на файловый вопрос модель видит файловые инструменты,
    но не задачные — короче промпт, меньше путаницы."""
    llm = FakeLLM(replies=["ответ по файлам"])
    tools = [_module_tool("list_files", "files"), _module_tool("create_task", "tasks")]
    orch = make_orchestrator(llm, db, tools=tools, topic_scoped_tools=True)
    await collect(orch, "покажи файлы в загрузках")
    sent = {t["function"]["name"] for t in (llm.seen_tools[0] or [])}
    assert "list_files" in sent
    assert "create_task" not in sent


async def test_topic_scoped_general_message_gets_no_tools(db: Database) -> None:
    """На общую реплику без темы инструменты не идут вовсе — самый лёгкий промпт."""
    llm = FakeLLM(replies=["привет!"])
    tools = [_module_tool("list_files", "files"), _module_tool("create_task", "tasks")]
    orch = make_orchestrator(llm, db, tools=tools, topic_scoped_tools=True)
    await collect(orch, "привет, как настроение")
    assert not (llm.seen_tools[0] or [])


class WidenProbeLLM:
    """Эмулирует строгий облачный бэкенд (Groq): если в наборе нет нужного
    инструмента — отклоняет ход (ToolChoiceError); когда набор расширили —
    вызывает инструмент, затем отвечает текстом."""

    def __init__(self) -> None:
        self.seen_tools: list[list[ToolSchema] | None] = []
        self._after_widen = 0

    def chat_runtime_is_local(self) -> bool:
        return False

    async def chat(self, role: Role, messages: list[ChatMessage]):  # pragma: no cover
        raise LLMError("не используется")

    async def stream(
        self, role: Role, messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None, tool_choice: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        from sba.llm.gateway import ToolChoiceError

        self.seen_tools.append(tools)
        names = {t["function"]["name"] for t in (tools or [])}
        if "create_task" not in names:
            raise ToolChoiceError(
                "attempted to call tool 'create_task' which was not in request.tools"
            )
        self._after_widen += 1
        if self._after_widen == 1:
            yield StreamEvent(tool_calls=[ToolCall(id="c1", name="create_task", arguments={})])
        else:
            yield StreamEvent(text="задача создана")


async def test_tool_choice_rejection_widens_tool_scope_and_recovers(db: Database) -> None:
    """topic_scoped урезал набор, модель позвала тул вне него, облако отклонило —
    оркестратор расширяет набор до всех и доводит ход (баг живой проверки)."""
    llm = WidenProbeLLM()
    tools = [_module_tool("list_files", "files"), _module_tool("create_task", "tasks")]
    orch = make_orchestrator(llm, db, tools=tools, topic_scoped_tools=True)
    reply = await collect(orch, "покажи файлы в загрузках")
    assert "задача создана" in reply
    # первый (узкий) набор — без create_task; после расширения — с ним
    assert "create_task" not in {t["function"]["name"] for t in (llm.seen_tools[0] or [])}
    assert "create_task" in {t["function"]["name"] for t in (llm.seen_tools[1] or [])}


async def test_tools_unscoped_by_default(db: Database) -> None:
    """Без флага поведение прежнее: модель видит все инструменты всегда."""
    llm = FakeLLM(replies=["ответ"])
    tools = [_module_tool("list_files", "files"), _module_tool("create_task", "tasks")]
    orch = make_orchestrator(llm, db, tools=tools)  # topic_scoped_tools=False по умолчанию
    await collect(orch, "покажи файлы в загрузках")
    sent = {t["function"]["name"] for t in (llm.seen_tools[0] or [])}
    assert {"list_files", "create_task"} <= sent


async def test_unknown_tool_error_fed_back_to_model(db: Database) -> None:
    llm = FakeLLM(
        replies=[[ToolCall(id="c1", name="ghost", arguments={})], "понял, инструмента нет"]
    )
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec([])]))
    assert text.endswith("понял, инструмента нет")
    _, second = llm.calls[1]
    assert any("не существует" in m.content for m in second if m.role == "tool")


async def test_text_json_tool_call_salvaged(db: Database) -> None:
    executed: list[str] = []
    llm = FakeLLM(
        replies=[
            '{"name": "probe", "arguments": {"value": "спасён"}}',  # JSON текстом
            "готово по данным инструмента",
        ]
    )
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec(executed)]))
    assert executed == ["спасён"]
    assert text.endswith("готово по данным инструмента")


async def test_json_array_tool_calls_not_leaked(db: Database) -> None:
    # случай живой проверки Sprint 8: после реального вызова 7B выдала массив
    # вызовов JSON-текстом — раньше он утекал сырым в чат
    executed: list[str] = []
    array = (
        '[{"name": "search_documents", "arguments": {"query": "Рейтингование"}}, '
        '{"name": "search_documents", "arguments": {"query": "Гранты"}}]'
    )
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "a"})],  # реальный вызов
            array,  # массив вызовов текстом — придерживаем, не показываем
            "Нашёл две заметки по теме.",  # после нуджа — человеческий ответ
        ]
    )
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec(executed)]))
    assert executed == ["a"]  # массив НЕ исполнен, только реальный вызов
    assert "search_documents" not in text  # сырой служебный формат не показан
    assert "Рейтингование" not in text
    assert "Нашёл две заметки" in text


async def test_persistent_json_text_refused(db: Database) -> None:
    executed: list[str] = []
    array = (
        '[{"name": "search_documents", "arguments": {"query": "A"}}, '
        '{"name": "search_documents", "arguments": {"query": "B"}}]'
    )
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "a"})],
            array,  # держим, один нудж
            array,  # снова служебный формат → честный отказ
        ]
    )
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec(executed)]))
    assert "служебным форматом" in text  # JSON_TEXT_REFUSAL
    assert "search_documents" not in text


async def test_single_json_text_not_shown_raw(db: Database) -> None:
    # одиночный вызов JSON-текстом по-прежнему спасается, но сырой JSON не виден
    executed: list[str] = []
    llm = FakeLLM(
        replies=['{"tool": "probe", "arguments": {"value": "z"}}', "готово"]
    )
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec(executed)]))
    assert executed == ["z"]
    assert '"tool"' not in text  # сырой JSON-ключ не показан
    assert "готово" in text


async def test_reply_starting_with_digit_streams(db: Database) -> None:
    # обычная проза (не с { или [) стримится как прежде — потоковость сохранена
    llm = FakeLLM(replies=["1) первый пункт, 2) второй"])
    assert await collect(make_orchestrator(llm, db)) == "1) первый пункт, 2) второй"


async def test_duplicate_tool_call_not_reexecuted(db: Database) -> None:
    executed: list[str] = []
    same = [ToolCall(id="c", name="probe", arguments={"value": "x"})]
    llm = FakeLLM(replies=[same, same, "ответ по данным"])
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec(executed)]))

    assert executed == ["x"]  # исполнен один раз, а не два
    assert text.endswith("ответ по данным")
    _, third_call = llm.calls[2]
    assert any("повторный вызов" in m.content for m in third_call if m.role == "tool")


async def test_iteration_limit_does_final_answer_pass(db: Database) -> None:
    """Шаги исчерпаны, но модель на финальном проходе БЕЗ инструментов
    отвечает по собранным данным — а не «не довёл дело до конца»."""
    call = [ToolCall(id="c", name="probe", arguments={"value": "x"})]
    llm = FakeLLM(replies=[call, call, "справка по собранным данным"])
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])], max_tool_iterations=2)
    )
    assert "справка по собранным данным" in text
    assert llm.seen_tools[-1] is None  # финальный проход шёл без инструментов


async def test_iteration_limit_fallback_when_still_no_answer(db: Database) -> None:
    """Если даже финальный проход не дал текста — честное сообщение."""
    call = [ToolCall(id="c", name="probe", arguments={"value": "x"})]
    llm = FakeLLM(replies=[call, call, call])  # и на финале снова вызов, не текст
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])], max_tool_iterations=2)
    )
    assert "Не удалось сформулировать" in text


# ── подтверждение destructive ────────────────────────────────────────────────


async def test_destructive_asks_confirmation_and_executes_on_yes(db: Database) -> None:
    executed: list[str] = []
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "файл.txt"})],
            "Удалил файл.txt",
        ]
    )
    orchestrator = make_orchestrator(
        llm, db, tools=[probe_spec(executed, risk=RiskLevel.DESTRUCTIVE)]
    )

    ask = await collect(orchestrator, "удали файл")
    assert "подтверждения" in ask
    assert executed == []  # ничего не выполнено до «да»

    answer = await collect(orchestrator, "да")
    assert executed == ["файл.txt"]
    # после подтверждения показываем результат инструмента напрямую и
    # останавливаемся — БЕЗ нового прохода LLM (иначе слабая модель срывается)
    assert "результат:файл.txt" in answer
    assert answer.startswith("✅")


async def test_confirmed_action_does_not_run_another_llm_round(db: Database) -> None:
    """Регрессия (живая проверка Sprint 8): после «да» слабая модель не должна
    получать новый круг agent-loop — иначе она вызывает выдуманные инструменты."""
    executed: list[str] = []
    # второй ответ модели — ловушка: если после подтверждения снова дёрнуть LLM,
    # выполнится левый вызов и попадёт в executed
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "нужный"})],
            [ToolCall(id="c2", name="probe", arguments={"value": "выдуманный"})],
        ]
    )
    orchestrator = make_orchestrator(
        llm, db, tools=[probe_spec(executed, risk=RiskLevel.DESTRUCTIVE)]
    )

    await collect(orchestrator, "удали")
    answer = await collect(orchestrator, "да")
    assert executed == ["нужный"]  # ровно один вызов, «выдуманный» не выполнился
    assert "выдуманный" not in answer


async def test_destructive_cancelled_on_no(db: Database) -> None:
    executed: list[str] = []
    llm = FakeLLM(replies=[[ToolCall(id="c1", name="probe", arguments={"value": "ф"})]])
    orchestrator = make_orchestrator(
        llm, db, tools=[probe_spec(executed, risk=RiskLevel.DESTRUCTIVE)]
    )

    await collect(orchestrator, "удали")
    answer = await collect(orchestrator, "нет")
    assert "отменено" in answer.lower()
    assert executed == []


async def test_expired_confirmation_is_not_fabricated(db: Database) -> None:
    """«да» после истечения срока → честный отказ, а НЕ фабрикация «готово»
    (живая проверка Sprint 8: ждал 24 мин, срок истёк, модель сочинила «удалил»)."""
    executed: list[str] = []
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "ф"})],
            "готово, удалил (сочинено)",  # ловушка: не должно дойти до модели
        ]
    )
    orchestrator = make_orchestrator(
        llm, db, tools=[probe_spec(executed, risk=RiskLevel.DESTRUCTIVE)]
    )

    await collect(orchestrator, "удали")
    # состарим ожидающее подтверждение далеко за пределы TTL
    orchestrator._pending[("u1", "cli")].created_at -= 10_000
    answer = await collect(orchestrator, "да")
    assert executed == []              # ничего не выполнено
    assert "истёк" in answer.lower()   # честный ответ, не выдуманное «готово»
    assert "сочинено" not in answer


async def test_unrelated_message_drops_pending(db: Database) -> None:
    executed: list[str] = []
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "ф"})],
            "обычный ответ",
        ]
    )
    orchestrator = make_orchestrator(
        llm, db, tools=[probe_spec(executed, risk=RiskLevel.DESTRUCTIVE)]
    )

    await collect(orchestrator, "удали")
    answer = await collect(orchestrator, "какая сегодня дата?")
    assert executed == []          # destructive так и не выполнен
    assert answer == "обычный ответ"


# ── принуждение к инструменту на нашей стороне (Ollama игнорирует required) ──


async def test_forced_tool_ignored_then_retried(db: Database) -> None:
    """Модель на «файловом» вопросе ответила текстом → текст скрыт, повтор
    со строгим втыком → вызов состоялся → нормальный ответ."""
    executed: list[str] = []
    llm = FakeLLM(
        replies=[
            "Нашёл файл отчёт.docx в Documents (сочинено)",
            [ToolCall(id="c1", name="probe", arguments={"value": "x"})],
            "готово по данным",
        ]
    )
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec(executed)]), "найди файл отчёт"
    )
    assert executed == ["x"]
    assert "сочинено" not in text  # выдуманный ответ пользователю не показан
    assert text.endswith("готово по данным")
    assert llm.seen_tool_choice[:2] == ["required", "required"]
    _, second = llm.calls[1]
    assert second[-1].role == "system"
    assert "не вызвав инструмент" in second[-1].content


async def test_forced_tool_refused_twice_hides_fabrication(db: Database) -> None:
    llm = FakeLLM(replies=["сочинение раз", "сочинение два"])
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])]), "найди файл отчёт"
    )
    assert "сочинение" not in text
    assert "⚠️" in text and "/new" in text


async def test_path_mention_without_tools_gets_audit_warning(db: Database) -> None:
    """Вне принудительных тем ответ с путями без единого вызова — растяжка."""
    llm = FakeLLM(replies=["Это лежит в C:\\Users\\x\\доклад.docx — я проверил"])
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])]), "а что по докладу?"
    )
    assert "доклад.docx" in text  # ответ показан…
    assert "/audit" in text  # …но с предупреждением о непроверенности


async def test_task_claim_without_tools_gets_warning(db: Database) -> None:
    """Модель заявляет «создана задача №…» без вызова инструмента — фабрикация."""
    llm = FakeLLM(replies=["Готово. Создана задача №a1b2c3 на письмо."])
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])]), "ну ладно, хорошо"
    )
    assert "не создано" in text.lower()
    assert "/tasks" in text


async def test_recurring_phrasing_forces_tool(db: Database) -> None:
    """«каждое утро…» и «напоминал» — задачные темы: принуждение к инструменту."""
    for phrase in (
        "каждое утро отправляй письмо",
        "хочу чтобы ты напоминал мне",
        "напомни, когда мне надо отправить письмо?",  # императив «напомни» тоже тема
    ):
        llm = FakeLLM(
            replies=[[ToolCall(id="c1", name="probe", arguments={"value": "x"})], "готово"]
        )
        orch = make_orchestrator(llm, db, tools=[probe_spec([])])
        await collect(orch, phrase)
        assert llm.seen_tool_choice[0] == "required", phrase


async def test_answer_after_real_tool_call_has_no_warning(db: Database) -> None:
    llm = FakeLLM(
        replies=[
            [ToolCall(id="c1", name="probe", arguments={"value": "x"})],
            "Файл отчёт.docx найден инструментом",
        ]
    )
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])]), "найди файл отчёт"
    )
    assert "/audit" not in text  # вызов был настоящим — растяжка молчит


async def test_info_question_triggers_file_nudge(db: Database) -> None:
    """«есть ли инфа про…» — вопрос о содержимом: нудж и принуждение включаются."""
    llm = FakeLLM(
        replies=[[ToolCall(id="c1", name="probe", arguments={"value": "x"})], "готово"]
    )
    await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])]),
        "есть ли инфа про абразивную губку?",
    )
    assert llm.seen_tool_choice[0] == "required"
    _, messages = llm.calls[0]
    assert any(
        m.role == "system" and "search_documents" in m.content for m in messages
    )

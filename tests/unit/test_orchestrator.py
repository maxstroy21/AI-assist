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
    async def chat(self, role: Role, messages: list[ChatMessage]):  # pragma: no cover
        raise LLMError("нет соединения")

    async def stream(
        self,
        role: Role,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        raise LLMError("нет соединения")
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


def make_orchestrator(
    llm,
    db: Database,
    history: list[HistoryEntry] | None = None,
    tools: list[ToolSpec] = (),
    **config,
) -> AgentOrchestrator:
    registry = ToolRegistry(AuditLog(db))
    for spec in tools:
        registry.register(spec)
    return AgentOrchestrator(
        gateway=llm,
        history=StubHistory(history or [HistoryEntry("user", "вопрос")]),
        registry=registry,
        config=AgentConfig(**config),
        timezone="Europe/Moscow",
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


async def test_llm_failure_becomes_friendly_message(db: Database) -> None:
    text = await collect(make_orchestrator(FailingLLM(), db))
    assert "⚠️" in text


async def test_file_question_gets_tool_nudge(db: Database) -> None:
    llm = FakeLLM(replies=["ок"])
    await collect(make_orchestrator(llm, db), "найди все файлы .log")
    _, messages = llm.calls[0]
    assert messages[-1].role == "system"
    assert "инструмент" in messages[-1].content


async def test_smalltalk_gets_no_nudge(db: Database) -> None:
    llm = FakeLLM(replies=["привет!"])
    await collect(make_orchestrator(llm, db), "привет, как дела?")
    _, messages = llm.calls[0]
    assert messages[-1].role == "user"


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


async def test_unknown_tool_error_fed_back_to_model(db: Database) -> None:
    llm = FakeLLM(
        replies=[[ToolCall(id="c1", name="ghost", arguments={})], "понял, инструмента нет"]
    )
    text = await collect(make_orchestrator(llm, db, tools=[probe_spec([])]))
    assert text.endswith("понял, инструмента нет")
    _, second = llm.calls[1]
    assert any("не существует" in m.content for m in second if m.role == "tool")


async def test_iteration_limit(db: Database) -> None:
    call = [ToolCall(id="c", name="probe", arguments={"value": "x"})]
    llm = FakeLLM(replies=[call, call, call])
    text = await collect(
        make_orchestrator(llm, db, tools=[probe_spec([])], max_tool_iterations=2)
    )
    assert "лимита шагов" in text


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
    assert answer.endswith("Удалил файл.txt")


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

from collections.abc import AsyncIterator

from sba.core.agent.orchestrator import AgentOrchestrator
from sba.core.history import HistoryEntry
from sba.core.types import IncomingMessage, Session, utcnow
from sba.infra.config import AgentConfig
from sba.llm.gateway import ChatMessage, LLMError, Role
from sba.llm.providers.fake import FakeLLM


class StubHistory:
    def __init__(self, entries: list[HistoryEntry]) -> None:
        self.entries = entries

    async def recent(self, conversation_id: str, limit: int) -> list[HistoryEntry]:
        return self.entries[-limit:]


class FailingLLM:
    async def chat(self, role: Role, messages: list[ChatMessage]):  # pragma: no cover
        raise LLMError("нет соединения")

    async def stream(self, role: Role, messages: list[ChatMessage]) -> AsyncIterator[str]:
        raise LLMError("нет соединения")
        yield ""  # unreachable, делает функцию генератором


def make_session() -> Session:
    now = utcnow()
    return Session(id="s1", user_id="u1", channel="cli", started_at=now, last_activity_at=now)


def make_orchestrator(llm, history: list[HistoryEntry], **config) -> AgentOrchestrator:
    return AgentOrchestrator(
        gateway=llm,
        history=StubHistory(history),
        config=AgentConfig(**config),
        timezone="Europe/Moscow",
    )


async def collect(orchestrator: AgentOrchestrator, text: str = "вопрос") -> str:
    reply = await orchestrator.process(
        IncomingMessage(user_id="u1", channel="cli", text=text), make_session()
    )
    assert not isinstance(reply, str)
    return "".join([d async for d in reply])


async def test_reply_streamed_from_llm() -> None:
    llm = FakeLLM(replies=["Привет! Чем помочь?"])
    history = [HistoryEntry("user", "привет")]
    assert await collect(make_orchestrator(llm, history)) == "Привет! Чем помочь?"


async def test_context_contains_system_and_history() -> None:
    llm = FakeLLM(replies=["ок"])
    history = [
        HistoryEntry("user", "меня зовут Макс"),
        HistoryEntry("assistant", "приятно познакомиться"),
        HistoryEntry("user", "как меня зовут?"),
    ]
    await collect(make_orchestrator(llm, history))

    role, messages = llm.calls[0]
    assert role == "chat"
    assert messages[0].role == "system"
    assert "Второй мозг" in messages[0].content
    assert [(m.role, m.content) for m in messages[1:]] == [
        ("user", "меня зовут Макс"),
        ("assistant", "приятно познакомиться"),
        ("user", "как меня зовут?"),
    ]


async def test_history_trimmed_to_budget_keeps_latest() -> None:
    llm = FakeLLM(replies=["ок"])
    history = [
        HistoryEntry("user", "старое " * 200),      # ~1400 символов
        HistoryEntry("assistant", "среднее " * 200),
        HistoryEntry("user", "свежий вопрос"),
    ]
    await collect(make_orchestrator(llm, history, history_budget_chars=100))

    _, messages = llm.calls[0]
    assert len(messages) == 2  # system + только свежайшее
    assert messages[-1].content == "свежий вопрос"


async def test_llm_failure_becomes_friendly_message() -> None:
    text = await collect(make_orchestrator(FailingLLM(), [HistoryEntry("user", "привет")]))
    assert "⚠️" in text
    assert "нет соединения" in text

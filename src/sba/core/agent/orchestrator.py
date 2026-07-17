"""Agent Orchestrator v1: контекст → LLM → потоковый ответ.

Инструментов пока нет (Sprint 2); контракт MessageProcessor тот же, что у эха,
поэтому Router и каналы не заметили подмены.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

from sba.core.agent.context import build_messages
from sba.core.history import HistoryReader
from sba.core.types import IncomingMessage, Reply, Session
from sba.infra.config import AgentConfig
from sba.llm.gateway import ChatMessage, LLMError, LLMGateway

log = structlog.get_logger(__name__)

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


class AgentOrchestrator:
    def __init__(
        self,
        gateway: LLMGateway,
        history: HistoryReader,
        config: AgentConfig,
        timezone: str,
    ) -> None:
        self._gateway = gateway
        self._history = history
        self._config = config
        self._tz = ZoneInfo(timezone)
        self._template = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    async def process(self, msg: IncomingMessage, session: Session) -> Reply:
        # история уже содержит текущее сообщение (Router сохраняет его до обработки)
        entries = await self._history.recent(session.id, self._config.history_max_messages)
        now = datetime.now(self._tz)
        system = self._template.format(
            now=now.strftime("%Y-%m-%d %H:%M, %A"), timezone=self._tz.key
        )
        messages = build_messages(system, entries, self._config.history_budget_chars)
        return self._reply_stream(messages)

    async def _reply_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        yielded_any = False
        try:
            async for delta in self._gateway.stream("chat", messages):
                yielded_any = True
                yield delta
        except LLMError as exc:
            log.error("llm_failed", error=str(exc))
            prefix = "\n\n" if yielded_any else ""
            yield (
                f"{prefix}⚠️ Не получилось обратиться к модели: {exc}\n"
                "Проверьте, что Ollama запущена (ollama ps)."
            )
            return
        if not yielded_any:
            yield "(модель вернула пустой ответ)"

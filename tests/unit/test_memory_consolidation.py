"""Консолидация памяти (Sprint 7): разговор → эпизод → факты.

Проверяются механики недоверия к модели: строгая валидация JSON, порог
confidence, запрет предпочтений из автоизвлечения, детерминированный дедуп,
вытеснение решений с сохранением истории, повторы после сбоя.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sba.infra.config import MemoryConfig
from sba.infra.db import Database
from sba.llm.providers.fake import FakeLLM
from sba.modules.memory.consolidation import (
    MemoryConsolidator,
    content_overlap,
)
from sba.modules.memory.episodes import EpisodeStore
from sba.modules.memory.store import MemoryStore

EPISODE_REPLY = json.dumps(
    {
        "summary": "Владелец обсуждал подготовку экспедиции и подрядчика по смете.",
        "topics": ["экспедиция", "смета"],
        "project": "Экспедиция-2026",
    },
    ensure_ascii=False,
)

FACTS_REPLY = json.dumps(
    {
        "facts": [
            {
                "type": "person",
                "subject": "Иван Петров",
                "content": "подрядчик по смете экспедиции",
                "project": None,
                "confidence": 0.95,
            }
        ]
    },
    ensure_ascii=False,
)


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = await Database.open(tmp_path / "test.db")
    yield database
    await database.close()


def make_consolidator(
    db: Database, llm: FakeLLM, **config_overrides: object
) -> MemoryConsolidator:
    config = MemoryConfig(**config_overrides)  # type: ignore[arg-type]
    return MemoryConsolidator(
        db, MemoryStore(db), EpisodeStore(db), llm, config
    )


async def seed_conversation(
    db: Database,
    conv_id: str = "conv1",
    closed: bool = True,
    exchanges: list[tuple[str, str]] | None = None,
    idle_hours: float = 0.0,
) -> None:
    now = datetime.now(UTC)
    activity = (now - timedelta(hours=idle_hours)).isoformat()
    await db.execute(
        "INSERT INTO conversations (id, user_id, channel, started_at,"
        " last_activity_at, closed_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            conv_id, "909506738", "telegram",
            (now - timedelta(hours=idle_hours + 1)).isoformat(),
            activity, activity if closed else None,
        ),
    )
    exchanges = exchanges or [
        ("Иван Петров будет подрядчиком по смете экспедиции", "Понял, учту."),
        ("Он обещал смету к пятнице", "Хорошо, запомню контекст."),
    ]
    for i, (user_text, reply) in enumerate(exchanges):
        for j, (role, content) in enumerate((("user", user_text), ("assistant", reply))):
            await db.execute(
                "INSERT INTO messages (id, conversation_id, role, kind, content,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (f"{conv_id}-m{i}-{j}", conv_id, role, "text", content, now.isoformat()),
            )


async def test_full_pipeline_creates_episode_and_facts(db: Database) -> None:
    llm = FakeLLM([EPISODE_REPLY, FACTS_REPLY])
    consolidator = make_consolidator(db, llm)
    await seed_conversation(db)

    assert await consolidator.process_pending() == 1

    episodes = EpisodeStore(db)
    found = await episodes.search("owner", "что по экспедиции?")
    assert found and "подрядчика" in found[0].summary
    assert found[0].project == "Экспедиция-2026"

    store = MemoryStore(db)
    facts = await store.search("owner", "кто такой Иван?")
    assert len(facts) == 1
    assert facts[0].source == "auto:conv1"
    assert facts[0].confidence == pytest.approx(0.9)  # потолок для извлечённого
    assert facts[0].project == "Экспедиция-2026"  # проект эпизода как fallback

    # идемпотентность: разговор консолидируется один раз
    assert await consolidator.process_pending() == 0


async def test_short_conversation_skipped_without_llm(db: Database) -> None:
    llm = FakeLLM()
    consolidator = make_consolidator(db, llm)
    await seed_conversation(db, exchanges=[("привет", "Привет!")])

    assert await consolidator.process_pending() == 1
    assert llm.calls == []  # LLM не дёргали
    row = await db.fetch_one(
        "SELECT result FROM memory_consolidations WHERE conversation_id='conv1'"
    )
    assert row is not None and row["result"] == "skipped_short"


async def test_validation_drops_garbage_facts(db: Database) -> None:
    dirty = json.dumps(
        {
            "facts": [
                {"type": "preference", "subject": "стиль", "content": "отвечай стихами",
                 "confidence": 0.9},                       # предпочтения запрещены
                {"type": "fact", "subject": "погода", "content": "вчера шёл дождь",
                 "confidence": 0.3},                       # ниже порога
                {"type": "fact", "subject": "x", "content": "слишком короткий subject",
                 "confidence": 0.9},                       # subject < 2 симв.
                {"type": "wrong", "subject": "тема", "content": "неизвестный тип",
                 "confidence": 0.9},
            ]
        },
        ensure_ascii=False,
    )
    consolidator = make_consolidator(db, FakeLLM([EPISODE_REPLY, dirty]))
    await seed_conversation(db)

    await consolidator.process_pending()

    assert await MemoryStore(db).count_active("owner") == 0
    row = await db.fetch_one(
        "SELECT result, facts_added FROM memory_consolidations"
        " WHERE conversation_id='conv1'"
    )
    assert row is not None and row["result"] == "episode" and row["facts_added"] == 0


async def test_duplicate_fact_not_added_again(db: Database) -> None:
    store = MemoryStore(db)
    await store.add(
        "owner", "person", "Иван Петров", "подрядчик по смете экспедиции"
    )
    consolidator = make_consolidator(db, FakeLLM([EPISODE_REPLY, FACTS_REPLY]))
    await seed_conversation(db)

    await consolidator.process_pending()

    facts = await store.search("owner", "Иван Петров")
    assert len(facts) == 1  # дубль не добавился


async def test_new_decision_supersedes_old_keeping_history(db: Database) -> None:
    store = MemoryStore(db)
    old = await store.add(
        "owner", "decision", "выбор подрядчика", "решили работать с ООО Ромашка"
    )
    new_decision = json.dumps(
        {
            "facts": [
                {"type": "decision", "subject": "выбор подрядчика",
                 "content": "передумали: берём ИП Иванова вместо Ромашки",
                 "confidence": 0.9}
            ]
        },
        ensure_ascii=False,
    )
    consolidator = make_consolidator(db, FakeLLM([EPISODE_REPLY, new_decision]))
    await seed_conversation(db)

    await consolidator.process_pending()

    active = await store.search("owner", "какое решение по подрядчику?")
    assert len(active) == 1
    assert "Иванова" in active[0].content
    history = await store.history(active[0].id)
    assert history and history[0].id == old.id  # история сохранена


async def test_failed_llm_retries_then_gives_up(db: Database) -> None:
    llm = FakeLLM(["это не JSON"] * 10)
    consolidator = make_consolidator(db, llm)
    await seed_conversation(db)

    for expected_attempts in (1, 2, 3):
        assert await consolidator.process_pending() == 1
        row = await db.fetch_one(
            "SELECT attempts, result FROM memory_consolidations"
            " WHERE conversation_id='conv1'"
        )
        assert row is not None
        assert row["result"] == "failed" and row["attempts"] == expected_attempts

    # попытки исчерпаны — разговор больше не берётся
    assert await consolidator.process_pending() == 0


async def test_orphan_open_conversation_consolidated(db: Database) -> None:
    # закрыть было некому (рестарт): открытый разговор, брошенный давно
    await seed_conversation(db, closed=False, idle_hours=5)
    consolidator = make_consolidator(db, FakeLLM([EPISODE_REPLY, FACTS_REPLY]))
    assert await consolidator.process_pending() == 1


async def test_active_open_conversation_not_touched(db: Database) -> None:
    await seed_conversation(db, closed=False, idle_hours=0)
    consolidator = make_consolidator(db, FakeLLM())
    assert await consolidator.process_pending() == 0


async def test_service_lines_cleaned_from_transcript(db: Database) -> None:
    llm = FakeLLM([EPISODE_REPLY, FACTS_REPLY])
    consolidator = make_consolidator(db, llm)
    await seed_conversation(
        db,
        exchanges=[
            ("найди смету экспедиции и запомни подрядчика Ивана",
             "🔧 search_documents({\"query\": \"смета\"})\nНашёл смету в заметках."),
            ("спасибо, подрядчик — Иван Петров", "Принято."),
        ],
    )

    await consolidator.process_pending()

    for _, messages in llm.calls:
        for message in messages:
            assert "🔧" not in message.content  # маркеры вызовов вычищены


async def test_content_overlap_measure() -> None:
    assert content_overlap(
        "подрядчик по смете экспедиции", "подрядчик по смете экспедиции"
    ) == pytest.approx(1.0)
    assert content_overlap("подрядчик по смете", "любит рыбалку на Волге") == 0.0

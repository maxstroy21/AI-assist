"""SQLite (WAL) + простые последовательные миграции через PRAGMA user_version."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
import structlog

log = structlog.get_logger(__name__)

# Каждый элемент — один шаг миграции; версия БД = количество применённых шагов.
# Правило: только добавлять в конец, никогда не редактировать применённые шаги.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE conversations (
        id               TEXT PRIMARY KEY,
        user_id          TEXT NOT NULL,
        channel          TEXT NOT NULL,
        started_at       TEXT NOT NULL,
        last_activity_at TEXT NOT NULL,
        closed_at        TEXT
    );
    CREATE INDEX idx_conversations_active
        ON conversations (user_id, channel) WHERE closed_at IS NULL;

    CREATE TABLE messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        role            TEXT NOT NULL,      -- user | assistant
        kind            TEXT NOT NULL,      -- text | voice | document | ...
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL
    );
    CREATE INDEX idx_messages_conversation ON messages (conversation_id, created_at);
    """,
    """
    CREATE TABLE audit_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        request_id TEXT,
        kind       TEXT NOT NULL,   -- tool_call | tool_result | confirmation_requested | ...
        name       TEXT NOT NULL,
        detail     TEXT NOT NULL,
        confirmed  INTEGER          -- NULL: не применимо; 0/1 для destructive
    );
    CREATE INDEX idx_audit_ts ON audit_log (ts);
    """,
    """
    CREATE TABLE memory_facts (
        id            TEXT PRIMARY KEY,
        user_id       TEXT NOT NULL,
        type          TEXT NOT NULL,   -- person | project | decision | preference | fact
        subject       TEXT NOT NULL,   -- кого/чего касается
        content       TEXT NOT NULL,   -- само знание
        source        TEXT NOT NULL,   -- explicit | msg:<id> | extracted (Sprint 7)
        confidence    REAL NOT NULL DEFAULT 1.0,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL,
        superseded_by TEXT,            -- вытеснен новым фактом (история сохраняется)
        retracted_at  TEXT             -- мягкое «забудь»
    );
    CREATE INDEX idx_memory_active ON memory_facts (user_id, type)
        WHERE superseded_by IS NULL AND retracted_at IS NULL;
    CREATE INDEX idx_memory_subject ON memory_facts (user_id, subject);

    CREATE VIRTUAL TABLE memory_fts USING fts5(subject, content, fact_id UNINDEXED);
    """,
    """
    CREATE TABLE rag_files (
        id           TEXT PRIMARY KEY,
        path         TEXT NOT NULL UNIQUE,
        content_hash TEXT,
        mtime        REAL,
        size         INTEGER,
        status       TEXT NOT NULL,     -- pending | indexed | failed
        error        TEXT,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        indexed_at   TEXT,
        updated_at   TEXT NOT NULL
    );

    CREATE TABLE rag_queue (
        path        TEXT PRIMARY KEY,   -- дедупликация: один файл — одна запись
        op          TEXT NOT NULL,      -- upsert | delete
        enqueued_at TEXT NOT NULL,
        attempts    INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE rag_chunks (
        id      TEXT PRIMARY KEY,       -- совпадает с id точки в Qdrant
        file_id TEXT NOT NULL REFERENCES rag_files(id) ON DELETE CASCADE,
        seq     INTEGER NOT NULL,
        text    TEXT NOT NULL,
        locator TEXT NOT NULL           -- «стр. 3», «раздел …», «фрагмент 2»
    );
    CREATE INDEX idx_rag_chunks_file ON rag_chunks (file_id);

    CREATE VIRTUAL TABLE rag_chunks_fts USING fts5(text, chunk_id UNINDEXED);
    """,
]


class Database:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    @classmethod
    async def open(cls, path: Path) -> Database:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        db = cls(conn)
        await db._migrate()
        return db

    async def _migrate(self) -> None:
        cursor = await self._conn.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        current = int(row[0]) if row else 0
        for version, script in enumerate(MIGRATIONS[current:], start=current + 1):
            await self._conn.executescript(script)
            await self._conn.execute(f"PRAGMA user_version={version}")
            await self._conn.commit()
            log.info("db_migrated", version=version)

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        await self._conn.execute(sql, params)
        await self._conn.commit()

    async def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> aiosqlite.Row | None:
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchone()

    async def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        cursor = await self._conn.execute(sql, params)
        return list(await cursor.fetchall())

    async def close(self) -> None:
        await self._conn.close()

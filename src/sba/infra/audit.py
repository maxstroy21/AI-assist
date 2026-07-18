"""Журнал действий (append-only): каждый вызов инструмента, подтверждения,
фоновые работы. Отвечает на вопрос «что ассистент делал и почему» (FR-9.3).
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from sba.infra.db import Database

DETAIL_MAX_CHARS = 2000


class AuditLog:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        kind: str,
        name: str,
        detail: str,
        confirmed: bool | None = None,
    ) -> None:
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        await self._db.execute(
            "INSERT INTO audit_log (ts, request_id, kind, name, detail, confirmed)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                request_id,
                kind,
                name,
                detail[:DETAIL_MAX_CHARS],
                None if confirmed is None else int(confirmed),
            ),
        )

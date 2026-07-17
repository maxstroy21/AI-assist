"""Структурированные логи (structlog) в stderr + request_id через contextvars.

request_id проставляется Router'ом на каждый входящий запрос и автоматически
попадает во все записи по пути его обработки (NFR-7).
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if fmt == "json"
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level]
        ),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )


def bind_request_id(request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()

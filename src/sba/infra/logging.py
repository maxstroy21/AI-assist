"""Структурированные логи (structlog) в stderr + request_id через contextvars.

request_id проставляется Router'ом на каждый входящий запрос и автоматически
попадает во все записи по пути его обработки (NFR-7).

Опционально дублируется в файл (logging.file) — чтобы при отладке ничего не
терялось: консоль прокручивается и на Windows подвисает от выделения текста,
а файл переживает и то, и другое. Пишем в оба потока сразу (tee).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO, Any

import structlog


class _Tee:
    """Пишет в несколько потоков разом (консоль + файл журнала)."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _open_log_file(path: str) -> IO[str]:
    """Открыть файл журнала на дозапись, построчная буферизация (flush по \\n)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.open("a", encoding="utf-8", buffering=1)


def setup_logging(level: str = "INFO", fmt: str = "console", file: str = "") -> None:
    # при записи в файл отключаем ANSI-цвета: иначе файл, который владелец
    # пришлёт в отладку, засоряется управляющими кодами и плохо читается
    renderer: Any = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=not file)
    )
    # Any: PrintLoggerFactory ждёт TextIO, но принимает любой объект с write/flush —
    # наш _Tee как раз такой (дублирование в консоль + файл)
    sink: Any = sys.stderr
    if file:
        sink = _Tee(sys.stderr, _open_log_file(file))
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
        logger_factory=structlog.PrintLoggerFactory(sink),
        cache_logger_on_first_use=True,
    )


def bind_request_id(request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()

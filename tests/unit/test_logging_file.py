"""Журнал в файл: логи дублируются в файл, не теряясь из консоли (Sprint 7 tooling)."""

import io
from pathlib import Path

import structlog

from sba.infra.logging import _Tee, setup_logging


def test_tee_writes_to_all_streams() -> None:
    a, b = io.StringIO(), io.StringIO()
    tee = _Tee(a, b)
    tee.write("привет")
    tee.flush()
    assert a.getvalue() == "привет"
    assert b.getvalue() == "привет"


def test_setup_logging_writes_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "sba.log"
    setup_logging(level="INFO", fmt="json", file=str(log_path))
    try:
        structlog.get_logger(__name__).info("proba_sobytiya", detail="значение")
    finally:
        # вернуть глобальную конфигурацию к дефолту, чтобы не влиять на другие тесты
        setup_logging()
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "proba_sobytiya" in content
    assert "значение" in content  # кириллица не экранируется (ensure_ascii=False)


def test_setup_logging_creates_parent_dirs(tmp_path: Path) -> None:
    log_path = tmp_path / "deep" / "nested" / "sba.log"
    setup_logging(level="INFO", fmt="json", file=str(log_path))
    try:
        structlog.get_logger(__name__).warning("ещё_событие")
    finally:
        setup_logging()
    assert log_path.exists()


def test_no_file_leaves_console_only(tmp_path: Path) -> None:
    # без file ничего не пишется на диск — регресс на дефолтное поведение
    setup_logging(level="INFO", fmt="console", file="")
    try:
        structlog.get_logger(__name__).info("only_console")
    finally:
        setup_logging()
    assert not list(tmp_path.iterdir())

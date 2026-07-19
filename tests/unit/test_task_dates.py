"""Парсер сроков: детерминированный слой и валидация LLM-fallback (Sprint 5)."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from sba.llm.providers.fake import FakeLLM
from sba.modules.tasks.dates import (
    ParsedWhen,
    WhenParseError,
    WhenParser,
    describe_rrule,
    parse_when_text,
)

TZ = ZoneInfo("Europe/Moscow")
# суббота, 2026-07-18, 12:00 московского времени
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=TZ)


def parse(text: str) -> ParsedWhen | None:
    return parse_when_text(text, NOW)


# ── детерминированный слой ───────────────────────────────────────────────────


def test_empty_means_no_due() -> None:
    assert parse("")  == ParsedWhen(kind="none")
    assert parse("без срока") == ParsedWhen(kind="none")


def test_tomorrow_with_time_keeps_timezone() -> None:
    parsed = parse("завтра в 9")
    assert parsed is not None and parsed.kind == "once"
    assert parsed.due == datetime(2026, 7, 19, 9, 0, tzinfo=TZ)
    assert parsed.due.utcoffset() == NOW.utcoffset()  # tz владельца, DoD Sprint 5


def test_tomorrow_evening_and_parts_of_day() -> None:
    parsed = parse("завтра вечером")
    assert parsed is not None and parsed.due is not None
    assert (parsed.due.day, parsed.due.hour) == (19, 19)
    parsed = parse("завтра в 9 вечера")
    assert parsed is not None and parsed.due is not None
    assert parsed.due.hour == 21


def test_today_default_hour_applied() -> None:
    parsed = parse_when_text("послезавтра", NOW, default_hour=10)
    assert parsed is not None
    assert parsed.due == datetime(2026, 7, 20, 10, 0, tzinfo=TZ)


def test_relative_hours_and_minutes() -> None:
    parsed = parse("через 2 часа")
    assert parsed is not None
    assert parsed.due == NOW.replace(hour=14)
    parsed = parse("через 30 минут")
    assert parsed is not None
    assert parsed.due == NOW.replace(hour=12, minute=30)
    parsed = parse("через неделю")
    assert parsed is not None
    assert parsed.due == datetime(2026, 7, 25, 9, 0, tzinfo=TZ)


def test_weekday_next_occurrence() -> None:
    # суббота → ближайший понедельник 20-го
    parsed = parse("в понедельник в 15:00")
    assert parsed is not None
    assert parsed.due == datetime(2026, 7, 20, 15, 0, tzinfo=TZ)


def test_same_weekday_goes_to_next_week() -> None:
    # сегодня суббота, время уже прошло → следующая суббота
    parsed = parse("в субботу в 9")
    assert parsed is not None
    assert parsed.due == datetime(2026, 7, 25, 9, 0, tzinfo=TZ)


def test_next_week_modifier() -> None:
    # «в следующий понедельник»: ближайший пн (20-е) уже на следующей неделе → он и есть
    parsed = parse("в следующий понедельник")
    assert parsed is not None
    assert parsed.due == datetime(2026, 7, 20, 9, 0, tzinfo=TZ)


def test_explicit_date_next_year_if_passed() -> None:
    parsed = parse("25.12 в 18:00")
    assert parsed is not None
    assert parsed.due == datetime(2026, 12, 25, 18, 0, tzinfo=TZ)
    parsed = parse("01.03")
    assert parsed is not None
    assert parsed.due == datetime(2027, 3, 1, 9, 0, tzinfo=TZ)  # 1 марта уже прошло


def test_month_name_date_not_read_as_time() -> None:
    parsed = parse("к 15 января")
    assert parsed is not None and parsed.kind == "once"
    assert parsed.due == datetime(2027, 1, 15, 9, 0, tzinfo=TZ)
    parsed = parse("25 декабря в 10:30")
    assert parsed is not None
    assert parsed.due == datetime(2026, 12, 25, 10, 30, tzinfo=TZ)


def test_time_only_rolls_to_tomorrow_if_passed() -> None:
    parsed = parse("в 9:00")
    assert parsed is not None
    assert parsed.due == datetime(2026, 7, 19, 9, 0, tzinfo=TZ)


def test_every_monday_is_correct_rrule() -> None:
    parsed = parse("каждый понедельник в 9")
    assert parsed is not None and parsed.kind == "recurring"
    assert parsed.rrule == "FREQ=WEEKLY;BYDAY=MO"
    assert parsed.due == datetime(2026, 7, 20, 9, 0, tzinfo=TZ)  # первое срабатывание


def test_recurring_variants() -> None:
    cases = {
        "каждый день": "FREQ=DAILY",
        "по будням": "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        "по пятницам": "FREQ=WEEKLY;BYDAY=FR",
        "каждую среду": "FREQ=WEEKLY;BYDAY=WE",
        "каждый месяц": "FREQ=MONTHLY",
        "ежегодно": "FREQ=YEARLY",
    }
    for text, rule in cases.items():
        parsed = parse(text)
        assert parsed is not None and parsed.kind == "recurring", text
        assert parsed.rrule == rule, text


def test_unknown_phrase_returns_none_for_llm() -> None:
    assert parse("когда освобожусь после отпуска") is None
    assert parse("каждые 2 недели") is None  # интервалы — забота LLM-слоя


# ── LLM-fallback: строгая валидация structured-ответа ────────────────────────


def make_parser(replies: list[str], clarify: float = 0.6) -> WhenParser:
    return WhenParser(FakeLLM(replies), TZ, clarify_confidence=clarify)


async def test_llm_once_is_validated_and_gets_timezone() -> None:
    parser = make_parser(
        ['{"kind": "once", "due": "2026-08-01 10:00", "confidence": 0.9, "question": null}']
    )
    parsed = await parser.parse("после отпуска, первого августа утром")
    assert parsed.kind == "once"
    assert parsed.due == datetime(2026, 8, 1, 10, 0, tzinfo=TZ)


async def test_llm_low_confidence_becomes_question() -> None:
    parser = make_parser(
        ['{"kind": "once", "due": "2026-08-01 10:00", "confidence": 0.3, "question": null}']
    )
    parsed = await parser.parse("ну там как обычно")
    assert parsed.kind == "unclear"
    assert parsed.question is not None


async def test_llm_unclear_passes_question_through() -> None:
    parser = make_parser(
        ['{"kind": "unclear", "due": null, "confidence": 0.9,'
         ' "question": "В этот понедельник или в следующий?"}']
    )
    parsed = await parser.parse("после праздников")
    assert parsed.kind == "unclear"
    assert parsed.question == "В этот понедельник или в следующий?"


async def test_llm_past_due_becomes_question() -> None:
    parser = make_parser(
        ['{"kind": "once", "due": "2020-01-01 10:00", "confidence": 0.95, "question": null}']
    )
    parsed = await parser.parse("вчерашняя дата")
    assert parsed.kind == "unclear"
    assert parsed.question is not None and "прошлом" in parsed.question


async def test_llm_recurring_rrule_is_validated() -> None:
    parser = make_parser(
        ['{"kind": "recurring", "due": null, "rrule": "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU",'
         ' "confidence": 0.9, "question": null}']
    )
    parsed = await parser.parse("раз в две недели по вторникам")
    assert parsed.kind == "recurring"
    assert parsed.rrule == "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU"
    assert parsed.due is not None and parsed.due > datetime.now(TZ)


async def test_llm_bad_rrule_raises() -> None:
    parser = make_parser(
        ['{"kind": "recurring", "rrule": "КАЖДЫЙ ВТОРНИК", "confidence": 0.9}']
    )
    with pytest.raises(WhenParseError):
        await parser.parse("раз в две недели")


async def test_llm_garbage_raises() -> None:
    parser = make_parser(["не могу разобрать"])
    with pytest.raises(WhenParseError):
        await parser.parse("что-то странное")


async def test_llm_json_inside_fences_is_extracted() -> None:
    parser = make_parser(
        ['```json\n{"kind": "none", "due": null, "confidence": 0.9, "question": null}\n```']
    )
    parsed = await parser.parse("просто заметка на будущее")
    assert parsed.kind == "none"


async def test_no_llm_raises() -> None:
    parser = WhenParser(None, TZ)
    with pytest.raises(WhenParseError):
        await parser.parse("когда освобожусь")


async def test_deterministic_skips_llm() -> None:
    fake = FakeLLM([])
    parser = WhenParser(fake, TZ)
    parsed = await parser.parse("завтра в 9")
    assert parsed.kind == "once"
    assert fake.calls == []  # LLM не трогали


# ── describe_rrule ───────────────────────────────────────────────────────────


def test_describe_rrule_common_cases() -> None:
    assert describe_rrule("FREQ=WEEKLY;BYDAY=MO") == "каждый понедельник"
    assert describe_rrule("FREQ=DAILY") == "каждый день"
    assert describe_rrule("FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR") == "по будням"
    assert describe_rrule("FREQ=MONTHLY") == "каждый месяц"
    # незнакомое — как есть, без вранья
    assert describe_rrule("FREQ=WEEKLY;INTERVAL=2;BYDAY=TU") == "FREQ=WEEKLY;INTERVAL=2;BYDAY=TU"

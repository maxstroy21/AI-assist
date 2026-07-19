"""Разбор сроков и повторений из естественного языка (Sprint 5).

Двухслойная схема — продолжение линии недоверия к модели (уроки Sprint 2/4):

1. Детерминированный разбор частых русских формулировок («завтра в 9»,
   «через 2 часа», «в пятницу», «каждый понедельник», «25.12») — быстро,
   без LLM и без фантазий.
2. Остальное — LLM (роль extraction, temperature 0) со structured-ответом,
   который ОБЯЗАТЕЛЬНО проходит валидацию: разбор даты, проверка RRULE через
   dateutil, «срок не в прошлом», порог уверенности. Низкая уверенность или
   неоднозначность → переспрос владельца, а не тихое угадывание.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Any, Literal

import structlog
from dateutil.relativedelta import relativedelta
from dateutil.rrule import rrulestr

from sba.llm.gateway import ChatMessage, LLMGateway

log = structlog.get_logger(__name__)

WhenKind = Literal["none", "once", "recurring", "unclear"]

WEEKDAY_NAMES = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
)
# стемы покрывают падежи и множественное число: «в среду», «по средам»
_WEEKDAY_STEMS: tuple[tuple[str, int, str], ...] = (
    ("понедельник", 0, "MO"),
    ("вторник", 1, "TU"),
    ("сред", 2, "WE"),
    ("четверг", 3, "TH"),
    ("пятниц", 4, "FR"),
    ("суббот", 5, "SA"),
    ("воскрес", 6, "SU"),
)
_STEMS_ALT = "|".join(stem for stem, _, _ in _WEEKDAY_STEMS)

_TIME_RE = re.compile(r"\b[вк]\s+(\d{1,2})(?:[:.](\d{2}))?\b(?:\s*(утра|дня|вечера|ночи))?")
_PART_OF_DAY = {"утром": (9, 0), "днём": (14, 0), "днем": (14, 0),
                "вечером": (19, 0), "ночью": (23, 0)}
_PART_RE = re.compile(r"\b(утром|днём|днем|вечером|ночью)\b")
_RELATIVE_RE = re.compile(
    r"\bчерез\s+(?:(\d+)\s*)?(минут|час|полчаса|д(?:ень|ня|ней)|недел|месяц)"
)
_WEEKDAY_RE = re.compile(rf"\b(?:в|во)\s+(?:(след\w*)\s+)?(?:(?:эт|ближайш)\w*\s+)?({_STEMS_ALT})")
_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{4}))?\b")
_MONTH_STEMS: tuple[tuple[str, int], ...] = (
    ("январ", 1), ("феврал", 2), ("март", 3), ("апрел", 4), ("ма[яе]", 5), ("июн", 6),
    ("июл", 7), ("август", 8), ("сентябр", 9), ("октябр", 10), ("ноябр", 11), ("декабр", 12),
)
_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})\s+({'|'.join(stem for stem, _ in _MONTH_STEMS)})\w*"
)
_DAY_WORDS: tuple[tuple[str, int], ...] = (("послезавтра", 2), ("завтра", 1), ("сегодня", 0))
_NO_DUE_RE = re.compile(r"без\s+срока|(?:убери|убрать|сними|снять)\s+срок")


@dataclass(frozen=True)
class ParsedWhen:
    kind: WhenKind
    due: datetime | None = None      # aware, tz владельца; у recurring — первый раз
    rrule: str | None = None         # RFC 5545 без префикса «RRULE:»
    question: str | None = None      # что переспросить при kind="unclear"


class WhenParseError(Exception):
    """Срок не разобран: LLM недоступна или вернула невалидный ответ."""


# ── слой 1: детерминированный разбор ─────────────────────────────────────────


def _extract_time(lowered: str) -> tuple[int, int] | None:
    m = _TIME_RE.search(lowered)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        suffix = m.group(3)
        if suffix in ("вечера", "дня") and hour < 12:
            hour += 12
        elif suffix == "ночи":
            hour = 0 if hour == 12 else (hour + 12 if hour >= 10 else hour)
        if hour <= 23 and minute <= 59:
            return hour, minute
    part = _PART_RE.search(lowered)
    if part:
        return _PART_OF_DAY[part.group(1)]
    return None


def _at(base: datetime, hm: tuple[int, int] | None, default_hour: int) -> datetime:
    hour, minute = hm if hm is not None else (default_hour, 0)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _recurring_rule(lowered: str) -> str | None:
    # интервалы («раз в две недели», «каждые 2 …») — забота LLM-слоя:
    # упрощённый разбор потерял бы INTERVAL и молча наврал бы с частотой
    if re.search(r"\bраз\s+в\b|\bкажд\w*\s+\d", lowered):
        return None
    if re.search(r"\bежедневн|\bкажд\w*\s+(?:день|утро)", lowered):
        return "FREQ=DAILY"
    if re.search(r"\bпо\s+будням|\bкажд\w*\s+будн", lowered):
        return "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR"
    m = re.search(rf"\b(?:кажд\w*|по)\s+({_STEMS_ALT})", lowered)
    if m:
        code = next(c for stem, _, c in _WEEKDAY_STEMS if stem == m.group(1))
        return f"FREQ=WEEKLY;BYDAY={code}"
    if re.search(r"\bеженедельн|\bкажд\w*\s+недел", lowered):
        return "FREQ=WEEKLY"
    if re.search(r"\bежемесячн|\bкажд\w*\s+месяц", lowered):
        return "FREQ=MONTHLY"
    if re.search(r"\bежегодн|\bкажд\w*\s+год", lowered):
        return "FREQ=YEARLY"
    return None


def next_occurrence(rule: str, dtstart: datetime, after: datetime) -> datetime | None:
    """Первое срабатывание правила строго после `after` (None — правило исчерпано)."""
    result: datetime | None = rrulestr(rule, dtstart=dtstart).after(after)
    return result


def _relative(lowered: str, now: datetime, hm: tuple[int, int] | None,
              default_hour: int) -> datetime | None:
    m = _RELATIVE_RE.search(lowered)
    if m is None:
        return None
    n = int(m.group(1) or 1)
    unit = m.group(2)
    if unit == "полчаса":
        return now + timedelta(minutes=30)
    if unit.startswith("минут"):
        return now + timedelta(minutes=n)
    if unit == "час":
        return now + timedelta(hours=n)
    if unit.startswith("д"):
        return _at(now + timedelta(days=n), hm, default_hour)
    if unit.startswith("недел"):
        return _at(now + timedelta(weeks=n), hm, default_hour)
    return _at(now + relativedelta(months=n), hm, default_hour)


def _weekday(lowered: str, now: datetime, hm: tuple[int, int] | None,
             default_hour: int) -> datetime | None:
    m = _WEEKDAY_RE.search(lowered)
    if m is None:
        return None
    index = next(i for stem, i, _ in _WEEKDAY_STEMS if stem == m.group(2))
    days_ahead = (index - now.weekday()) % 7
    candidate = _at(now + timedelta(days=days_ahead), hm, default_hour)
    if candidate <= now:
        candidate += timedelta(days=7)
    # «следующий понедельник» = понедельник следующей недели, даже если
    # ближайший — на этой; на стыке недель неоднозначность решает LLM-слой
    if m.group(1) and candidate.isocalendar()[:2] == now.isocalendar()[:2]:
        candidate += timedelta(days=7)
    return candidate


def _build_date(now: datetime, day: int, month: int, year: int | None,
                hm: tuple[int, int] | None, default_hour: int) -> datetime | None:
    try:
        candidate = _at(now.replace(year=year or now.year, month=month, day=day),
                        hm, default_hour)
    except ValueError:
        return None
    if year is None and candidate <= now:
        try:
            candidate = candidate.replace(year=candidate.year + 1)
        except ValueError:  # 29 февраля
            return None
    return candidate


def _mask(lowered: str, m: re.Match[str] | None) -> str:
    if m is None:
        return lowered
    return lowered[: m.start()] + " " * (m.end() - m.start()) + lowered[m.end():]


def _explicit_date(lowered: str, now: datetime, hm: tuple[int, int] | None,
                   default_hour: int) -> datetime | None:
    # маскируем фрагмент времени, чтобы «в 9.30» не читалось как «9 марта»
    masked = _mask(lowered, _TIME_RE.search(lowered))
    for m in _DATE_RE.finditer(masked):
        candidate = _build_date(
            now, int(m.group(1)), int(m.group(2)),
            int(m.group(3)) if m.group(3) else None, hm, default_hour,
        )
        if candidate is not None:
            return candidate
    return None


def _month_date(m: re.Match[str], now: datetime, hm: tuple[int, int] | None,
                default_hour: int) -> datetime | None:
    month = next(i for stem, i in _MONTH_STEMS if re.fullmatch(stem, m.group(2)))
    return _build_date(now, int(m.group(1)), month, None, hm, default_hour)


def parse_when_text(text: str, now: datetime, default_hour: int = 9) -> ParsedWhen | None:
    """Детерминированный разбор срока. None — формулировка не распознана
    (решает LLM-слой). `now` обязан быть aware в часовом поясе владельца."""
    lowered = " ".join(text.lower().split())
    if not lowered or _NO_DUE_RE.search(lowered):
        return ParsedWhen(kind="none")
    month_match = _MONTH_RE.search(lowered)
    # «к 15 января» — это дата, а не время «к 15»
    hm = _extract_time(_mask(lowered, month_match))

    rule = _recurring_rule(lowered)
    if rule is not None:
        dtstart = _at(now, hm, default_hour)
        return ParsedWhen(kind="recurring", rrule=rule, due=next_occurrence(rule, dtstart, now))

    due = _relative(lowered, now, hm, default_hour)
    if due is not None:
        return ParsedWhen(kind="once", due=due)

    due = _weekday(lowered, now, hm, default_hour)
    if due is not None:
        return ParsedWhen(kind="once", due=due)

    for word, offset in _DAY_WORDS:
        if word in lowered:
            return ParsedWhen(kind="once", due=_at(now + timedelta(days=offset), hm, default_hour))

    if month_match is not None:
        due = _month_date(month_match, now, hm, default_hour)
        if due is not None:
            return ParsedWhen(kind="once", due=due)

    due = _explicit_date(lowered, now, hm, default_hour)
    if due is not None:
        return ParsedWhen(kind="once", due=due)

    if hm is not None:
        # «в 15:00» без даты — но только если кроме времени во фразе ничего
        # нет: «первого августа утром» должен уйти в LLM целиком
        leftover = _PART_RE.sub(" ", _mask(lowered, _TIME_RE.search(lowered)))
        if not re.search(r"[a-zа-яё]{3,}|\d", leftover):
            candidate = _at(now, hm, default_hour)
            if candidate <= now:
                candidate += timedelta(days=1)
            return ParsedWhen(kind="once", due=candidate)
    return None


# ── слой 2: LLM-fallback со строгой валидацией ───────────────────────────────

_PROMPT = """Сейчас {now} ({weekday}), часовой пояс {tz}.
Определи срок задачи из фразы владельца: «{text}»

Ответь ОДНИМ JSON-объектом без пояснений и без markdown:
{{"kind": "once", "due": "2026-01-15 09:00", "rrule": null, "confidence": 0.9, "question": null}}

Поля:
- kind: "once" — конкретный срок; "recurring" — повторение; "none" — срока нет; \
"unclear" — неоднозначно.
- due: "YYYY-MM-DD HH:MM" (обязательно для once; для recurring — первое срабатывание) либо null.
- rrule: правило повторения RFC 5545, только для recurring, например "FREQ=WEEKLY;BYDAY=MO", \
либо null.
- confidence: уверенность в разборе от 0 до 1.
- question: при kind "unclear" — короткий уточняющий вопрос владельцу, иначе null."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_PAST_SLACK = timedelta(minutes=5)


def _extract_json(text: str) -> dict[str, Any]:
    m = _JSON_RE.search(text)
    if m is None:
        raise WhenParseError(f"в ответе модели нет JSON: {text[:120]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise WhenParseError(f"невалидный JSON от модели: {exc}") from exc
    if not isinstance(data, dict):
        raise WhenParseError("модель вернула не JSON-объект")
    return data


def _parse_due(raw: object, tz: tzinfo) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise WhenParseError(f"невалидное поле due: {raw!r}")
    try:
        due = datetime.fromisoformat(raw.strip().replace(" ", "T"))
    except ValueError as exc:
        raise WhenParseError(f"невалидная дата due: {raw!r}") from exc
    return due.replace(tzinfo=tz) if due.tzinfo is None else due


class WhenParser:
    """Фасад разбора сроков: детерминированный слой + LLM-fallback."""

    def __init__(
        self,
        llm: LLMGateway | None,
        timezone: tzinfo,
        default_hour: int = 9,
        clarify_confidence: float = 0.6,
    ) -> None:
        self._llm = llm
        self._tz = timezone
        self._default_hour = default_hour
        self._clarify_confidence = clarify_confidence

    async def parse(self, text: str) -> ParsedWhen:
        now = datetime.now(self._tz)
        parsed = parse_when_text(text, now, self._default_hour)
        if parsed is not None:
            return parsed
        if self._llm is None:
            raise WhenParseError("роль extraction не настроена в models.yaml")
        prompt = _PROMPT.format(
            now=now.strftime("%Y-%m-%d %H:%M"),
            weekday=WEEKDAY_NAMES[now.weekday()],
            tz=str(self._tz),
            text=" ".join(text.split()),
        )
        result = await self._llm.chat("extraction", [ChatMessage(role="user", content=prompt)])
        parsed = self._validate(_extract_json(result.text), text, now)
        log.info("when_parsed_by_llm", text=text, kind=parsed.kind,
                 due=str(parsed.due), rrule=parsed.rrule)
        return parsed

    def _validate(self, data: dict[str, Any], text: str, now: datetime) -> ParsedWhen:
        kind = data.get("kind")
        question = data.get("question")
        default_question = (
            f"Не смог однозначно понять срок «{' '.join(text.split())}» — "
            "уточните, пожалуйста (например: «завтра в 15:00» или «каждый понедельник»)."
        )
        if kind == "none":
            return ParsedWhen(kind="none")
        if kind == "unclear":
            q = question if isinstance(question, str) and question.strip() else default_question
            return ParsedWhen(kind="unclear", question=q)
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self._clarify_confidence:
            return ParsedWhen(kind="unclear", question=default_question)

        if kind == "once":
            due = _parse_due(data.get("due"), self._tz)
            if due < now - _PAST_SLACK:
                return ParsedWhen(
                    kind="unclear",
                    question=f"Получился срок в прошлом ({due.strftime('%d.%m.%Y %H:%M')}) — "
                    "уточните, пожалуйста, когда нужно.",
                )
            return ParsedWhen(kind="once", due=due)

        if kind == "recurring":
            raw_rule = data.get("rrule")
            if not isinstance(raw_rule, str) or not raw_rule.strip():
                raise WhenParseError("kind=recurring без правила rrule")
            rule = raw_rule.strip().removeprefix("RRULE:").upper()
            dtstart = (
                _parse_due(data.get("due"), self._tz)
                if data.get("due")
                else now.replace(hour=self._default_hour, minute=0, second=0, microsecond=0)
            )
            try:
                first = next_occurrence(rule, dtstart, now)
            except (ValueError, KeyError) as exc:
                raise WhenParseError(f"невалидный RRULE {rule!r}: {exc}") from exc
            return ParsedWhen(kind="recurring", rrule=rule, due=first)

        raise WhenParseError(f"неизвестный kind от модели: {kind!r}")


def describe_rrule(rule: str) -> str:
    """Правило повторения → по-русски (частые случаи; иначе — как есть)."""
    parts = dict(p.split("=", 1) for p in rule.split(";") if "=" in p)
    if parts.get("INTERVAL") not in (None, "1"):
        return rule
    freq, byday = parts.get("FREQ"), parts.get("BYDAY", "")
    if freq == "DAILY":
        return "каждый день"
    if freq == "WEEKLY":
        if byday == "MO,TU,WE,TH,FR":
            return "по будням"
        names = {"MO": "каждый понедельник", "TU": "каждый вторник", "WE": "каждую среду",
                 "TH": "каждый четверг", "FR": "каждую пятницу", "SA": "каждую субботу",
                 "SU": "каждое воскресенье"}
        if byday in names:
            return names[byday]
        if not byday:
            return "каждую неделю"
    if freq == "MONTHLY" and not byday:
        return "каждый месяц"
    if freq == "YEARLY":
        return "каждый год"
    return rule

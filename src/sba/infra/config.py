"""Config Service: default.yaml ⊕ local.yaml ⊕ переменные окружения → Pydantic.

Ошибка конфига = отказ старта с внятным сообщением (ConfigError).
Переопределение из окружения: SBA__SECTION__KEY=value, напр. SBA__LOGGING__LEVEL=DEBUG.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

ENV_PREFIX = "SBA__"


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppConfig(_Strict):
    data_dir: Path = Path("./data")
    timezone: str = "Europe/Moscow"

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"неизвестный часовой пояс: {v!r}") from exc
        return v


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["console", "json"] = "console"


class SessionConfig(_Strict):
    idle_timeout_minutes: int = 60

    @field_validator("idle_timeout_minutes")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("idle_timeout_minutes должен быть > 0")
        return v


class ChannelToggle(_Strict):
    enabled: bool = False


class TelegramChannelConfig(_Strict):
    enabled: bool = False
    token: str = ""
    allowed_user_ids: list[int] = []


class AgentConfig(_Strict):
    processor: Literal["llm", "echo"] = "llm"
    history_max_messages: int = 16
    history_budget_chars: int = 4000
    max_tool_iterations: int = 5


class FilesConfig(_Strict):
    allowed_roots: list[Path] = []
    max_list_entries: int = 50
    max_read_chars: int = 4000


class ChannelsConfig(_Strict):
    cli: ChannelToggle = ChannelToggle(enabled=True)
    telegram: TelegramChannelConfig = TelegramChannelConfig()


class RagConfig(_Strict):
    enabled: bool = True
    sources: list[Path] = []            # папки с документами (задаются в local.yaml)
    include_extensions: list[str] = [".pdf", ".docx", ".md", ".txt"]
    max_file_mb: int = 50
    scan_interval_minutes: float = 2.0
    dialog_cooldown_seconds: float = 90.0  # пауза индексации после сообщения в диалоге
    chunk_chars: int = 1800             # ~450 токенов на фрагмент
    chunk_overlap_chars: int = 200
    search_top_k: int = 5
    snippet_chars: int = 700            # длина цитаты в результате поиска
    embed_batch: int = 8

    @field_validator("include_extensions")
    @classmethod
    def _dotted_lower(cls, v: list[str]) -> list[str]:
        for ext in v:
            if not ext.startswith("."):
                raise ValueError(f"расширение должно начинаться с точки: {ext!r}")
        return [ext.lower() for ext in v]


class TasksConfig(_Strict):
    enabled: bool = True
    list_limit: int = 15          # сколько задач показывают /tasks и search_tasks
    default_hour: int = 9         # час срока, если во фразе есть день, но нет времени
    clarify_confidence: float = 0.6  # ниже — переспрашиваем вместо создания задачи

    @field_validator("default_hour")
    @classmethod
    def _valid_hour(cls, v: int) -> int:
        if not 0 <= v <= 23:
            raise ValueError("default_hour должен быть в диапазоне 0–23")
        return v


class SchedulerConfig(_Strict):
    tick_seconds: float = 15.0            # период проверки созревших джобов
    misfire_grace_minutes: float = 240.0  # skip-джобы (сводка): доставить не позже grace

    @field_validator("tick_seconds", "misfire_grace_minutes")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("значение должно быть > 0")
        return v


class MorningBriefConfig(_Strict):
    enabled: bool = True
    time: str = "08:30"

    @field_validator("time")
    @classmethod
    def _valid_time(cls, v: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError(f"время сводки должно быть в формате ЧЧ:ММ, получено {v!r}")
        return v


class RemindersConfig(_Strict):
    enabled: bool = True
    snooze_minutes: int = 180        # «⏰ Позже» без уточнения времени
    followup_hours: float = 24.0     # «пока не сделано»: период повторного напоминания
    morning_brief: MorningBriefConfig = MorningBriefConfig()

    @field_validator("snooze_minutes", "followup_hours")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("значение должно быть > 0")
        return v


class MemoryConfig(_Strict):
    enabled: bool = True
    # семантический recall памяти держит эмбеддинг-модель (bge-m3) в RAM почти
    # на каждое сообщение. На машине с дефицитом памяти это дорого: false →
    # память ищет по словам (FTS, как в Sprint 3), эмбеддинг-модель грузится
    # только под поиск по документам. Поиск по документам от этого не страдает.
    semantic: bool = True


class ModulesConfig(_Strict):
    memory: MemoryConfig = MemoryConfig()
    rag: RagConfig = RagConfig()
    tasks: TasksConfig = TasksConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    reminders: RemindersConfig = RemindersConfig()


class LLMBehaviorConfig(_Strict):
    # 0 — отключить прогрев; иначе пинг каждые N минут держит модель в RAM
    keep_warm_minutes: float = 4.0


class Config(_Strict):
    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    session: SessionConfig = SessionConfig()
    agent: AgentConfig = AgentConfig()
    llm: LLMBehaviorConfig = LLMBehaviorConfig()
    files: FilesConfig = FilesConfig()
    modules: ModulesConfig = ModulesConfig()
    channels: ChannelsConfig = ChannelsConfig()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: некорректный YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: ожидался YAML-словарь, получено {type(data).__name__}")
    return data


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """SBA__A__B=x → {"a": {"b": x}}; значения парсятся как YAML-скаляры."""
    result: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        keys = [k.lower() for k in name.removeprefix(ENV_PREFIX).split("__") if k]
        if not keys:
            continue
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return result


def load_config(config_dir: Path, environ: dict[str, str] | None = None) -> Config:
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise ConfigError(f"не найден {default_path}")
    data = _load_yaml(default_path)

    local_path = config_dir / "local.yaml"
    if local_path.exists():
        data = _deep_merge(data, _load_yaml(local_path))

    data = _deep_merge(data, _env_overrides(environ if environ is not None else dict(os.environ)))

    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(problems) from exc

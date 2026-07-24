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
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

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
    # путь к файлу журнала: логи дублируются туда (плюс к консоли), чтобы при
    # отладке ничего не терялось. Пусто — только консоль. Пример: 'data/logs/sba.log'
    file: str = ""


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


class WebChannelConfig(_Strict):
    """Локальный web-чат (Sprint 9): FastAPI + WebSocket на машине владельца."""

    enabled: bool = False
    host: str = "127.0.0.1"   # менять только осознанно (напр. Tailscale-интерфейс):
                              # аутентификации в web-чате нет, защита — локальность
    port: int = 8765
    history_messages: int = 30  # сколько последних сообщений показать при открытии

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port должен быть в диапазоне 1–65535")
        return v

    @field_validator("history_messages")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("history_messages должен быть > 0")
        return v


class AgentConfig(_Strict):
    processor: Literal["llm", "echo"] = "llm"
    history_max_messages: int = 16
    history_budget_chars: int = 4000
    max_tool_iterations: int = 5
    # true — слать модели только инструменты по теме сообщения (файлы/задачи/
    # память), а не все сразу. Короче промпт → быстрее ответ на CPU и меньше
    # путаницы у слабой модели. На общие реплики («привет») инструменты не идут
    topic_scoped_tools: bool = False


class FilesConfig(_Strict):
    allowed_roots: list[Path] = []
    max_list_entries: int = 50
    max_read_chars: int = 4000


class ChannelsConfig(_Strict):
    cli: ChannelToggle = ChannelToggle(enabled=True)
    telegram: TelegramChannelConfig = TelegramChannelConfig()
    web: WebChannelConfig = WebChannelConfig()


class RagConfig(_Strict):
    enabled: bool = True
    sources: list[Path] = []            # папки с документами (задаются в local.yaml)
    include_extensions: list[str] = [".pdf", ".docx", ".md", ".txt", ".xlsx", ".xlsm"]
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
    # автопамять (Sprint 7): закрытые разговоры фоново сворачиваются в эпизоды,
    # из них извлекаются факты (роли summarize/extraction). Работает в паузах
    # диалога, чтобы не конкурировать с ответами за CPU
    auto_extract: bool = True
    min_confidence: float = 0.7        # извлечённые факты ниже порога отбрасываются
    max_facts_per_session: int = 5     # защита от мусора: не больше фактов с разговора
    check_interval_seconds: float = 600.0   # период поиска неконсолидированных разговоров
    dialog_cooldown_seconds: float = 90.0   # тишина в диалоге перед LLM-вызовами
    llm_timeout_seconds: float = 180.0      # таймаут одного LLM-шага консолидации

    @field_validator(
        "check_interval_seconds", "dialog_cooldown_seconds", "llm_timeout_seconds",
        "max_facts_per_session",
    )
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("значение должно быть > 0")
        return v

    @field_validator("min_confidence")
    @classmethod
    def _unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("min_confidence должен быть в диапазоне 0–1")
        return v


class FileOpsConfig(_Strict):
    enabled: bool = True
    # false — СУХОЙ ПРОГОН (защита самого опасного функционала, риск Sprint 8):
    # операции move/copy/rename/archive только планируются и журналируются,
    # файловая система не меняется. Включать в local.yaml после недели проверки
    # планов владельцем. Поиск дубликатов/версий — только чтение, работает всегда
    execute: bool = False
    archive_subdir: str = "_архив"   # подпапка архива внутри разбираемой папки
    max_batch_files: int = 200       # потолок файлов одной операции архивирования
    scan_limit_files: int = 20_000   # потолок обхода при поиске дубликатов/версий
    time_budget_seconds: float = 30.0  # бюджет времени одного сканирования
    max_groups: int = 10             # сколько групп дубликатов/версий показывать

    @field_validator("max_batch_files", "scan_limit_files", "time_budget_seconds", "max_groups")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("значение должно быть > 0")
        return v

    @field_validator("archive_subdir")
    @classmethod
    def _plain_name(cls, v: str) -> str:
        if not v.strip() or "/" in v or "\\" in v or v in {".", ".."}:
            raise ValueError("archive_subdir должен быть простым именем папки")
        return v.strip()


class McpServerEntry(_Strict):
    """Внешний MCP-сервер (Sprint 9): его инструменты попадают в общий Tool
    Registry и подчиняются тем же уровням риска (ADR-7, ADR-10)."""

    name: str                          # короткое имя: git, fetch, calendar…
    enabled: bool = True
    # транспорт: либо локальный процесс (stdio), либо URL (streamable HTTP)
    command: str = ""                  # напр. 'uvx' или 'npx'
    args: list[str] = []               # напр. ['mcp-server-git']
    env: dict[str, str] = {}           # переменные окружения процесса
    url: str = ""                      # напр. 'http://localhost:8080/mcp'
    # риск инструментов сервера; по умолчанию консервативно destructive —
    # каждый вызов чужого кода требует подтверждения, пока владелец явно
    # не понизил риск в конфиге (docs/04 §15)
    risk: Literal["read", "write", "destructive"] = "destructive"
    tool_risks: dict[str, Literal["read", "write", "destructive"]] = {}
    connect_timeout_seconds: float = 20.0

    @field_validator("name")
    @classmethod
    def _plain_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
            raise ValueError(
                f"имя MCP-сервера должно быть из букв/цифр/дефисов, получено {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _one_transport(self) -> McpServerEntry:
        if bool(self.command) == bool(self.url):
            raise ValueError(
                f"MCP-сервер {self.name!r}: укажите ровно одно из command (stdio) "
                "или url (HTTP)"
            )
        return self


class McpConfig(_Strict):
    # подключаемые внешние серверы; пустой список = MCP-клиент выключен
    servers: list[McpServerEntry] = []
    # наш MCP-сервер (python -m sba.mcp.server): какие МОДУЛИ экспортировать.
    # Именно список модулей, а не имён инструментов: новый инструмент
    # существующего модуля утекает наружу сам собой (docs/06 §Sprint 9)
    export_modules: list[str] = ["rag", "memory", "tasks"]

    @model_validator(mode="after")
    def _unique_names(self) -> McpConfig:
        names = [s.name for s in self.servers]
        if len(names) != len(set(names)):
            raise ValueError("имена MCP-серверов должны быть уникальны")
        return self


class ModulesConfig(_Strict):
    memory: MemoryConfig = MemoryConfig()
    rag: RagConfig = RagConfig()
    tasks: TasksConfig = TasksConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    reminders: RemindersConfig = RemindersConfig()
    fileops: FileOpsConfig = FileOpsConfig()


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
    mcp: McpConfig = McpConfig()


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

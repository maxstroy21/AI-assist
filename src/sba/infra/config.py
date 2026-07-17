"""Config Service: default.yaml ⊕ local.yaml ⊕ переменные окружения → Pydantic.

Ошибка конфига = отказ старта с внятным сообщением (ConfigError).
Переопределение из окружения: SBA__SECTION__KEY=value, напр. SBA__LOGGING__LEVEL=DEBUG.
"""

from __future__ import annotations

import os
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


class ChannelsConfig(_Strict):
    cli: ChannelToggle = ChannelToggle(enabled=True)
    telegram: ChannelToggle = ChannelToggle(enabled=False)


class Config(_Strict):
    app: AppConfig = AppConfig()
    logging: LoggingConfig = LoggingConfig()
    session: SessionConfig = SessionConfig()
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

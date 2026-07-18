"""Схема и загрузка config/models.yaml (роли моделей → рантаймы)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from sba.infra.config import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(_Strict):
    kind: Literal["openai_compatible"]
    base_url: str
    api_key: str = ""


class RoleConfig(_Strict):
    runtime: str
    model: str
    temperature: float | None = None
    # native — модель умеет tool calling сама; json — инструкция в промпте
    # и разбор JSON-ответа (страховка для моделей без tool calling)
    tool_mode: Literal["native", "json"] = "native"


class ModelsConfig(_Strict):
    runtimes: dict[str, RuntimeConfig]
    roles: dict[str, RoleConfig]

    @model_validator(mode="after")
    def _roles_reference_known_runtimes(self) -> ModelsConfig:
        for role_name, role in self.roles.items():
            if role.runtime not in self.runtimes:
                raise ValueError(
                    f"роль {role_name!r} ссылается на неизвестный рантайм {role.runtime!r}"
                )
        if "chat" not in self.roles:
            raise ValueError("обязательная роль 'chat' не настроена")
        return self


def load_models_config(path: Path) -> ModelsConfig:
    if not path.exists():
        raise ConfigError(
            f"не найден {path} — файл ролей моделей обязателен при agent.processor=llm"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ModelsConfig.model_validate(data)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: некорректный YAML: {exc}") from exc
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: {problems}") from exc

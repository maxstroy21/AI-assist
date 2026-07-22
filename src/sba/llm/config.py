"""Схема и загрузка config/models.yaml (роли моделей → рантаймы)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from sba.infra.config import ConfigError

# подстановка секретов из окружения: ${ИМЯ} в значениях models.yaml → значение
# переменной. Так ключ облачного провайдера не попадает в git-файл (лежит в env
# владельца). Работаем по разобранной структуре — ссылки в комментариях игнорируются
_ENV_REF = re.compile(r"\$\{(\w+)\}")


def _expand_env(node: Any) -> Any:
    if isinstance(node, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            value = os.environ.get(name)
            if value is None:
                raise ConfigError(
                    f"config/models.yaml ссылается на переменную окружения {name}, "
                    f"но она не задана. Задайте её (Windows: setx {name} \"ваш_ключ\", "
                    "затем перезапустите PowerShell) или уберите ссылку из файла"
                )
            return value

        return _ENV_REF.sub(repl, node)
    if isinstance(node, dict):
        return {key: _expand_env(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_expand_env(item) for item in node]
    return node


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(_Strict):
    # openai_compatible — Ollama/LM Studio и т.п. (base_url обязателен);
    # anthropic — облачный Claude через официальный SDK (base_url не нужен,
    # авторизация — ключ ANTHROPIC_API_KEY или вход `ant auth login` под аккаунтом)
    kind: Literal["openai_compatible", "anthropic"]
    base_url: str = ""
    api_key: str = ""

    @model_validator(mode="after")
    def _base_url_for_openai(self) -> RuntimeConfig:
        if self.kind == "openai_compatible" and not self.base_url:
            raise ValueError("base_url обязателен для kind=openai_compatible")
        return self


class RoleConfig(_Strict):
    runtime: str
    model: str
    temperature: float | None = None
    # потолок токенов ответа. Anthropic требует его явно (провайдер подставит
    # 4096, если не задан); для openai_compatible None = «дефолт модели»
    max_tokens: int | None = None
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
        data = _expand_env(yaml.safe_load(path.read_text(encoding="utf-8")))
        return ModelsConfig.model_validate(data)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: некорректный YAML: {exc}") from exc
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"{path}: {problems}") from exc

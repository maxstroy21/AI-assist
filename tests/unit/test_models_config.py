from pathlib import Path

import pytest

from sba.infra.config import ConfigError
from sba.llm.config import load_models_config

VALID = """
runtimes:
  ollama: {kind: openai_compatible, base_url: "http://localhost:11434/v1"}
roles:
  chat: {runtime: ollama, model: "qwen2.5:7b-instruct", temperature: 0.7}
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path: Path) -> None:
    config = load_models_config(write(tmp_path, VALID))
    assert config.roles["chat"].model == "qwen2.5:7b-instruct"


def test_unknown_runtime_rejected(tmp_path: Path) -> None:
    text = VALID.replace("runtime: ollama, model", "runtime: ghost, model")
    with pytest.raises(ConfigError, match="ghost"):
        load_models_config(write(tmp_path, text))


def test_chat_role_required(tmp_path: Path) -> None:
    text = VALID.replace("chat:", "summarize:")
    with pytest.raises(ConfigError, match="chat"):
        load_models_config(write(tmp_path, text))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="не найден"):
        load_models_config(tmp_path / "models.yaml")


def test_repo_models_yaml_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    # реальный файл ссылается на ${GROQ_API_KEY} — в CI переменной нет,
    # подставляем фиктивную: тест проверяет схему файла, а не ключ
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-dummy")
    config = load_models_config(Path(__file__).parents[2] / "config" / "models.yaml")
    # роль chat настроена (рантайм — облако или локальный ollama, меняется)
    assert "chat" in config.roles
    # эмбеддинги обязаны остаться локальными: у облачных провайдеров их нет,
    # а поиск по документам не должен зависеть от облака (Local First)
    assert config.roles["embedding"].runtime == "ollama"


def test_anthropic_runtime_needs_no_base_url(tmp_path: Path) -> None:
    text = """
runtimes:
  cloud: {kind: anthropic}
roles:
  chat: {runtime: cloud, model: "claude-sonnet-5", max_tokens: 4096}
"""
    config = load_models_config(write(tmp_path, text))
    assert config.runtimes["cloud"].kind == "anthropic"
    assert config.roles["chat"].max_tokens == 4096


def test_openai_runtime_requires_base_url(tmp_path: Path) -> None:
    text = """
runtimes:
  ollama: {kind: openai_compatible}
roles:
  chat: {runtime: ollama, model: "qwen2.5:7b-instruct"}
"""
    with pytest.raises(ConfigError, match="base_url"):
        load_models_config(write(tmp_path, text))


DEEPSEEK = """
runtimes:
  deepseek:
    kind: openai_compatible
    base_url: https://api.deepseek.com
    api_key: "${DEEPSEEK_API_KEY}"
roles:
  chat: {runtime: deepseek, model: "deepseek-chat"}
"""


def test_env_var_in_api_key_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-123")
    config = load_models_config(write(tmp_path, DEEPSEEK))
    assert config.runtimes["deepseek"].api_key == "sk-secret-123"


def test_missing_env_var_raises_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        load_models_config(write(tmp_path, DEEPSEEK))

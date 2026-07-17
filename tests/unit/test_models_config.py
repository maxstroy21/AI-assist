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


def test_repo_models_yaml_is_valid() -> None:
    config = load_models_config(Path(__file__).parents[2] / "config" / "models.yaml")
    assert config.roles["chat"].runtime == "ollama"

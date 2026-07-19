from pathlib import Path

import pytest

from sba.infra.config import ConfigError, load_config


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_defaults_load(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  timezone: Europe/Moscow\n")
    config = load_config(tmp_path, environ={})
    assert config.logging.level == "INFO"
    assert config.channels.cli.enabled is True


def test_memory_semantic_toggle(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  timezone: Europe/Moscow\n")
    default = load_config(tmp_path, environ={})
    assert default.modules.memory.semantic is True  # по умолчанию — по смыслу
    write(tmp_path / "local.yaml", "modules:\n  memory:\n    semantic: false\n")
    tuned = load_config(tmp_path, environ={})
    assert tuned.modules.memory.semantic is False
    assert tuned.modules.memory.enabled is True  # память остаётся, меняется только поиск


def test_local_overrides_default(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "logging:\n  level: INFO\n")
    write(tmp_path / "local.yaml", "logging:\n  level: DEBUG\n")
    config = load_config(tmp_path, environ={})
    assert config.logging.level == "DEBUG"


def test_env_overrides_everything(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "logging:\n  level: INFO\n")
    config = load_config(tmp_path, environ={"SBA__LOGGING__LEVEL": "ERROR"})
    assert config.logging.level == "ERROR"


def test_invalid_value_fails_with_clear_error(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "session:\n  idle_timeout_minutes: -5\n")
    with pytest.raises(ConfigError, match="idle_timeout_minutes"):
        load_config(tmp_path, environ={})


def test_unknown_key_rejected(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  data_dirr: ./oops\n")
    with pytest.raises(ConfigError, match="data_dirr"):
        load_config(tmp_path, environ={})


def test_invalid_timezone_rejected(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  timezone: Mars/Olympus\n")
    with pytest.raises(ConfigError, match="часовой пояс"):
        load_config(tmp_path, environ={})


def test_missing_default_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"default\.yaml"):
        load_config(tmp_path, environ={})


def test_repo_default_yaml_is_valid() -> None:
    config = load_config(Path(__file__).parents[2] / "config", environ={})
    assert config.channels.cli.enabled is True

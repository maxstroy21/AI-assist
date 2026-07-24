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


def test_memory_auto_extract_defaults(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  timezone: Europe/Moscow\n")
    config = load_config(tmp_path, environ={})
    memory = config.modules.memory
    assert memory.auto_extract is True
    assert memory.min_confidence == 0.7
    assert memory.max_facts_per_session == 5


def test_memory_min_confidence_validated(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "modules:\n  memory:\n    min_confidence: 1.5\n")
    with pytest.raises(ConfigError, match="min_confidence"):
        load_config(tmp_path, environ={})


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


def test_web_channel_defaults_and_validation(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  timezone: Europe/Moscow\n")
    config = load_config(tmp_path, environ={})
    web = config.channels.web
    assert web.enabled is False
    assert (web.host, web.port) == ("127.0.0.1", 8765)
    write(tmp_path / "local.yaml", "channels:\n  web:\n    port: 70000\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path, environ={})


def test_mcp_defaults(tmp_path: Path) -> None:
    write(tmp_path / "default.yaml", "app:\n  timezone: Europe/Moscow\n")
    config = load_config(tmp_path, environ={})
    assert config.mcp.servers == []
    assert config.mcp.export_modules == ["rag", "memory", "tasks"]


def test_mcp_server_entry_parsed(tmp_path: Path) -> None:
    write(
        tmp_path / "default.yaml",
        "mcp:\n  servers:\n    - name: git\n      command: uvx\n"
        "      args: [mcp-server-git]\n      risk: read\n"
        "      tool_risks:\n        git_commit: destructive\n",
    )
    config = load_config(tmp_path, environ={})
    server = config.mcp.servers[0]
    assert (server.name, server.command, server.risk) == ("git", "uvx", "read")
    assert server.tool_risks == {"git_commit": "destructive"}


def test_mcp_entry_requires_exactly_one_transport(tmp_path: Path) -> None:
    write(
        tmp_path / "default.yaml",
        "mcp:\n  servers:\n    - name: bad\n      command: uvx\n"
        "      url: http://localhost:1234/mcp\n",
    )
    with pytest.raises(ConfigError):
        load_config(tmp_path, environ={})
    write(
        tmp_path / "default.yaml",
        "mcp:\n  servers:\n    - name: empty\n",
    )
    with pytest.raises(ConfigError):
        load_config(tmp_path, environ={})


def test_mcp_duplicate_server_names_rejected(tmp_path: Path) -> None:
    write(
        tmp_path / "default.yaml",
        "mcp:\n  servers:\n"
        "    - name: git\n      command: uvx\n"
        "    - name: git\n      command: npx\n",
    )
    with pytest.raises(ConfigError):
        load_config(tmp_path, environ={})

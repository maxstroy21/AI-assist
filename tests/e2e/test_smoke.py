"""Smoke: приложение стартует целиком и echo-диалог проходит через REPL."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_app_starts_and_echoes(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        f"app:\n  data_dir: {tmp_path / 'data'}\n", encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "sba", "--config-dir", str(config_dir)],
        input="привет, ассистент\n/quit\n",
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[echo] привет, ассистент" in proc.stdout


def test_bad_config_fails_cleanly(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "session:\n  idle_timeout_minutes: -1\n", encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, "-m", "sba", "--config-dir", str(config_dir)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 2
    assert "Ошибка конфигурации" in proc.stderr

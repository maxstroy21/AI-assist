"""Бот переживает занятый Qdrant (Sprint 9, живая проверка).

Embedded Qdrant однопроцессный: второй запущенный экземпляр бота (или
незавершённый предыдущий) держит файлы. Открытие падает блокировкой —
но это не должно ронять весь ассистент: он обязан стартовать без поиска
по документам, а память уйти на FTS. Раньше падал стеной трейсбека.
"""

from __future__ import annotations

from pathlib import Path

from sba.app import App
from sba.infra.vectors import VectorStore, VectorStoreError

MODELS = (
    "runtimes:\n  ollama:\n    kind: openai_compatible\n"
    "    base_url: http://localhost:11434/v1\n    api_key: ''\n"
    "roles:\n"
    "  chat: {runtime: ollama, model: qwen2.5:3b-instruct}\n"
    "  extraction: {runtime: ollama, model: qwen2.5:3b-instruct}\n"
    "  embedding: {runtime: ollama, model: bge-m3}\n"
)


def _write_config(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                f"app:\n  data_dir: {tmp_path / 'data'}",
                "modules:",
                "  rag:\n    enabled: true\n    sources: ['" + str(tmp_path) + "']",
                "  memory:\n    enabled: true\n    semantic: true\n    auto_extract: false",
                "  reminders:\n    enabled: false",
                "channels:\n  cli:\n    enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "models.yaml").write_text(MODELS, encoding="utf-8")
    return config_dir


async def test_app_starts_when_qdrant_locked(tmp_path, monkeypatch):
    config_dir = _write_config(tmp_path)

    def locked(_path: Path) -> VectorStore:
        raise VectorStoreError("файлы Qdrant уже заняты другим процессом")

    monkeypatch.setattr(VectorStore, "open", classmethod(lambda cls, path: locked(path)))

    # ключевое: не бросает, приложение собирается целиком
    app = await App.create(config_dir)
    try:
        assert app.vectors is None      # индекс не открылся
        assert app.indexer is None      # индексатор не поднят (нет rag-сервиса)
    finally:
        await app.shutdown()


async def test_app_starts_normally_when_qdrant_free(tmp_path):
    """Контроль: без блокировки rag поднимается как обычно."""
    config_dir = _write_config(tmp_path)
    app = await App.create(config_dir)
    try:
        assert app.vectors is not None
        assert app.indexer is not None
    finally:
        await app.shutdown()

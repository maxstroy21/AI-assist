# Этап 4. Структура проекта

Структура отражает архитектурные границы: `core` не знает о модулях, модули не знают
о каналах, всё внешнее — адаптеры. Нарушения границ ловит `import-linter` в CI.

```
second-brain/
├── pyproject.toml              # uv/poetry; единый пакет `sba`
├── README.md
├── docs/                       # этот архитектурный проект + будущие ADR
│   └── adr/                    # новые решения — по одному файлу ADR-NNN.md
│
├── config/
│   ├── default.yaml            # полный конфиг с дефолтами (в git)
│   ├── local.yaml              # переопределения пользователя (в .gitignore)
│   └── models.yaml             # роли моделей → рантайм/имя/параметры
│
├── src/sba/
│   ├── __main__.py             # python -m sba → запуск всего приложения
│   ├── app.py                  # композиция: DI-контейнер, feature flags,
│   │                           #   порядок старта/останова модулей
│   │
│   ├── core/                   # ЯДРО — не импортирует modules/ и channels/
│   │   ├── events.py           #   Event Bus (async pub/sub) + типы событий
│   │   ├── router.py           #   Message Router: нормализация, сессии, доставка
│   │   ├── agent/
│   │   │   ├── orchestrator.py #   agent loop (итерации, риск, подтверждения)
│   │   │   ├── context.py      #   Context Builder (бюджеты токенов на секции)
│   │   │   └── prompts/        #   системные промпты (файлы, не строки в коде)
│   │   ├── tools/
│   │   │   ├── registry.py     #   Tool Registry
│   │   │   └── spec.py         #   ToolSpec, уровни риска
│   │   └── types.py            #   IncomingMessage, OutgoingMessage, Session...
│   │
│   ├── llm/                    # LLM Gateway
│   │   ├── gateway.py          #   роли → провайдеры, retry, учёт токенов
│   │   ├── providers/          #   openai_compat.py, sentence_tf.py, whisper.py
│   │   └── structured.py       #   structured output + JSON-fallback + retry
│   │
│   ├── channels/               # адаптеры каналов; знают ТОЛЬКО core.router
│   │   ├── telegram/           #   aiogram: handlers, voice, файлы, кнопки
│   │   ├── webchat/            #   FastAPI + WebSocket, localhost-only
│   │   └── cli/                #   REPL для разработки
│   │
│   ├── modules/                # МОДУЛИ — каждый: interface.py + tools.py
│   │   ├── memory/             #   interface, store (SQLite), extraction,
│   │   │   └── ...             #   consolidation, retrieval, tools
│   │   ├── rag/                #   retrieval (hybrid+rerank), qdrant client, tools
│   │   ├── indexer/            #   watcher, queue, extractors/ (pdf, docx, xlsx,
│   │   │   └── ...             #   md, obsidian, code, ocr), chunker, catalog
│   │   ├── tasks/              #   модель задач, NL-парсинг дат (через LLM), tools
│   │   ├── reminders/          #   правила (RRULE), доставка, snooze, follow-up
│   │   ├── fileops/            #   безопасные файловые операции, undo-журнал,
│   │   │   └── ...             #   дубликаты, архивирование
│   │   └── scheduler/          #   APScheduler-обёртка, регистрация джобов
│   │
│   ├── mcp/
│   │   ├── client_hub.py       # подключение внешних MCP-серверов из конфига,
│   │   │                       #   маппинг их тулов в Tool Registry
│   │   └── server.py           # экспорт наших инструментов как MCP-сервер
│   │                           #   (отдельный entrypoint: python -m sba.mcp.server)
│   │
│   ├── infra/
│   │   ├── db.py               # SQLite: соединения, миграции (alembic/просто DDL)
│   │   ├── config.py           # Config Service: yaml + env, Pydantic Settings
│   │   ├── logging.py          # structlog: JSON-логи, request_id через contextvars
│   │   ├── audit.py            # append-only журнал действий
│   │   └── backup.py           # бэкап/restore/verify всех хранилищ
│   │
│   └── security/
│       ├── acl.py              # whitelist пользователей, whitelisted корни ФС
│       └── http.py             # единый HTTP-клиент с allowlist хостов
│
├── data/                       # всё состояние (в .gitignore); путь — в конфиге
│   ├── sba.db                  #   SQLite
│   ├── qdrant/                 #   embedded-хранилище Qdrant
│   ├── files_cache/            #   извлечённый текст, OCR-кэш
│   └── backups/
│
├── tests/
│   ├── unit/                   # по модулю на пакет; LLM — фейк-провайдер
│   ├── integration/            # router→agent→tools на fake-LLM со сценариями
│   └── e2e/                    # smoke: старт приложения, консольный диалог
│
└── scripts/
    ├── install_service_windows.ps1   # NSSM / Task Scheduler
    ├── install_service_linux.sh      # systemd unit
    ├── reindex.py                    # ручная полная переиндексация
    └── restore_backup.py             # восстановление + проверка
```

## Назначение ключевых папок

| Папка | Роль | Правило зависимостей |
|-------|------|----------------------|
| `core/` | Оркестрация, роутинг, реестр инструментов, событийная шина | Не импортирует `modules/`, `channels/`; знает только `llm/`, `infra/` |
| `llm/` | Единственная точка контакта с моделями | Никто, кроме `core` и модулей, не зовёт LLM напрямую |
| `channels/` | Тонкие адаптеры: перевод «канал ⇄ IncomingMessage/OutgoingMessage» | Импортируют только `core.router`, `core.types` |
| `modules/*` | Вся предметная логика; каждый модуль: `interface.py` (Protocol), `tools.py` (ToolSpec-декларации), своё хранилище | Межмодульные связи — только через interface или события |
| `mcp/` | Мост MCP в обе стороны | Клиент-хаб маппит внешние тулы в Registry; сервер экспортирует внутренние |
| `infra/` | Кросс-срезы: конфиг, БД, логи, аудит, бэкапы | Не содержит предметной логики |
| `security/` | ACL и сетевая политика | Используется каналами, fileops и HTTP-клиентом |
| `data/` | Всё состояние в одном месте | Одна папка = один объект бэкапа; переносимость на другую машину копированием |

## Как добавляется новая возможность (проверка расширяемости)

| Что добавляем | Что делаем | Что НЕ трогаем |
|---------------|-----------|----------------|
| Новый инструмент | Функция + ToolSpec в `tools.py` модуля | core, каналы |
| Новый источник данных | Экстрактор в `indexer/extractors/` + запись в конфиг | rag, core |
| Новый канал (голос, web) | Пакет в `channels/` с адаптером к Router | всё остальное |
| Новая модель/рантайм | Строки в `models.yaml` | код |
| Внешняя интеграция | MCP-сервер в конфиге `mcp.servers` | код |
| Новый агент (напр. «ресёрчер») | Профиль агента: свой промпт + свой поднабор тулов в конфиге | core (оркестратор общий) |

**Следующий документ:** [04-services.md](04-services.md).

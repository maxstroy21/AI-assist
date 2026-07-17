# Этап 5. Основные сервисы

Для каждого сервиса: зона ответственности, публичный интерфейс (то, что видят другие
модули — сигнатуры упрощены), хранилище, зависимости, события.

---

## 1. LLM Gateway (`llm/`)

**Ответственность:** единственная точка обращения к моделям. Роли (`chat`, `extraction`,
`summarize`, `embedding`, `rerank`, `stt`) → конкретные провайдеры из `models.yaml`.
Retry, таймауты, бюджет токенов, structured output (Pydantic + JSON-fallback для моделей
без нативного tool calling), учёт использования.

```python
class LLMGateway(Protocol):
    async def chat(role, messages, tools=None, stream=False) -> ChatResult
    async def structured(role, messages, schema: type[T]) -> T          # с retry
    async def embed(texts: list[str]) -> list[Vector]
    async def rerank(query, candidates) -> list[Scored]
    async def transcribe(audio: bytes) -> str
```

**Хранилище:** `llm_usage` (учёт вызовов/токенов). **Зависимости:** нет (лист дерева).

---

## 2. Message Router (`core/router.py`)

**Ответственность:** нормализация входящих из любого канала в `IncomingMessage`
(текст уже транскрибирован, документ уже сохранён во входящую папку), управление
сессиями (одна активная сессия на пользователя, таймаут неактивности → закрытие
с событием `session_closed`), доставка исходящих в нужный канал (ответы — в канал-источник,
напоминания — во все активные каналы доставки).

```python
class Router(Protocol):
    async def handle_incoming(msg: IncomingMessage) -> None      # канал → ядро
    async def deliver(out: OutgoingMessage) -> None              # ядро → канал(ы)
    def register_channel(name, channel: ChannelAdapter) -> None
```

**Хранилище:** `conversations`, `messages`. **События:** публикует `message_received`, `session_closed`.

---

## 3. Agent Orchestrator (`core/agent/`)

**Ответственность:** agent loop (см. диаграмму в 02-architecture §2.3): построение
контекста, вызовы LLM, исполнение tool calls, обработка `destructive`-подтверждений,
лимиты (итерации/токены/время), streaming ответа в канал. Профили агентов
(default, morning-brief, researcher) — промпт + поднабор инструментов из конфига.

**Зависимости:** LLM Gateway, Tool Registry, Context Builder, Router (для подтверждений).
**События:** публикует `message_processed` (вход + ответ + использованные тулзы).

---

## 4. Context Builder (`core/agent/context.py`)

**Ответственность:** сборка промпта с бюджетами токенов по секциям:
system + процедурная память (всегда) → релевантные факты памяти (recall по запросу) →
история диалога (последние N + rolling summary) → результаты инструментов.
Режет по приоритету при переполнении, помечает контент документов как данные
(анти prompt-injection).

---

## 5. Tool Registry (`core/tools/`)

**Ответственность:** регистрация ToolSpec от модулей и MCP-хаба, выдача агенту
списка доступных инструментов (фильтры: feature flags, профиль агента, контекст),
исполнение вызова с валидацией аргументов, записью в audit и маршрутизацией
`destructive` через подтверждение.

```python
class ToolRegistry(Protocol):
    def register(spec: ToolSpec) -> None
    def available(profile: AgentProfile) -> list[ToolSpec]
    async def execute(call: ToolCall, ctx: ExecContext) -> ToolResult
```

---

## 6. Memory Service (`modules/memory/`)

**Ответственность:** 4 слоя памяти (см. 02-architecture §3): CRUD типизированных фактов,
семантический recall, извлечение фактов из диалогов (по `message_processed`, роль
`extraction`), консолидация сессий в эпизоды (по `session_closed` и ночью), вытеснение
противоречий (`superseded_by`), мягкое забывание.

```python
class MemoryService(Protocol):
    async def remember(fact: NewFact, source: Source) -> FactId
    async def recall(query: str, project_id=None, k=8) -> list[MemoryItem]
    async def forget(query_or_id) -> list[FactId]
    async def preferences() -> list[Preference]        # для Context Builder
```

**Инструменты:** `remember_fact`, `recall`, `forget`.
**Хранилище:** SQLite `memory_facts`, `episodes`, `preferences` + Qdrant-коллекция `memory`.

---

## 7. RAG Service (`modules/rag/`)

**Ответственность:** retrieval поверх Qdrant: гибридный поиск (dense + sparse, RRF),
payload-фильтры (тип, папка, даты, проект, теги), опциональный rerank, сборка
цитируемых фрагментов (путь + локатор). Управление коллекциями и named vectors
(миграция эмбеддера).

```python
class RAGService(Protocol):
    async def search(query, filters: SearchFilters = None, k=8) -> list[Passage]
    async def similar(doc_id, k=10) -> list[DocRef]                 # похожие документы
    async def upsert_chunks(doc: DocMeta, chunks: list[Chunk]) -> None   # для индексатора
    async def delete_doc(doc_id) -> None
```

**Инструменты:** `search_documents`, `read_document` (полный текст из кэша), `find_similar_documents`.

---

## 8. File Indexer (`modules/indexer/`)

**Ответственность:** наблюдение за источниками (watchdog + периодический полный скан),
очередь с приоритетами и дедупликацией (SQLite, переживает рестарт), извлечение текста
(PDF/DOCX/XLSX/MD/код/OCR), Obsidian-обогащение (wikilinks, теги, frontmatter), чанкинг,
эмбеддинг, upsert в RAG, ведение каталога `files` (path, hash, mtime, статус).
Пауза при активном диалоге, тихие часы для тяжёлых работ.

**События:** подписан на `file_changed` (от watcher) и `reindex_requested`; публикует `document_indexed`.

---

## 9. Task Manager (`modules/tasks/`)

**Ответственность:** модель задач (title, notes, due, RRULE-повторение, project_id,
status, source-ссылка на сообщение), парсинг естественного языка дат/повторений через
LLM (роль `extraction` → structured output: `due | rrule | trigger-условие`),
поиск задач (SQL + семантика), предложение слияния дубликатов (кандидаты по векторной
близости → подтверждение пользователем).

**Инструменты:** `create_task`, `update_task`, `complete_task`, `search_tasks`, `merge_tasks`.
**Хранилище:** SQLite `tasks`. **События:** публикует `task_created`, `task_completed`, `task_overdue`.

---

## 10. Reminder Engine (`modules/reminders/`)

**Ответственность:** напоминания как отдельная сущность (может ссылаться на задачу):
одноразовые, повторяющиеся (RRULE), условные follow-up («если не сделал — снова через 2 дня»).
Регистрирует срабатывания в Scheduler; при срабатывании формирует `OutgoingMessage`
с кнопками (✅ сделано / ⏰ позже / ✖ отменить) и отдаёт Router'у. «Умный момент»
без срока: LLM-эвристика по типу задачи + предпочтениям (утро для планирования,
рабочие часы для рабочих).

**Инструменты:** `create_reminder`, `snooze_reminder`, `cancel_reminder`.
**Хранилище:** SQLite `reminders`, `reminder_log`.

---

## 11. File Ops (`modules/fileops/`)

**Ответственность:** безопасные операции ФС строго внутри whitelisted корней:
move/copy/rename/create/archive с транзакционным undo-журналом; поиск дубликатов
(hash) и почти-дубликатов (векторная близость через RAG); старые версии
(эвристики имени `_v2, (1), копия` + similar + mtime).

**Инструменты:** `list_files`, `move_file`, `copy_file`, `rename_file`, `create_file`,
`archive_files`, `find_duplicates`, `find_old_versions`, `undo_file_operation`.
Уровни риска: массовые/перезаписывающие — `destructive`.

---

## 12. Scheduler (`modules/scheduler/`)

**Ответственность:** обёртка над APScheduler (SQLite job store): cron-джобы
(ночная консолидация памяти, переиндексация, бэкап, утренняя сводка, ежемесячная
проверка восстановления) и одноразовые (срабатывания напоминаний). Джобы —
только публикация событий в Bus (сам ничего не исполняет — слабая связанность).

```python
class Scheduler(Protocol):
    def schedule_once(at: datetime, event: Event) -> JobId
    def schedule_rrule(rrule: str, event: Event) -> JobId
    def cancel(job_id) -> None
```

---

## 13. Telegram Gateway (`channels/telegram/`)

**Ответственность:** aiogram 3, long polling. Whitelist user_id (чужие — тишина + лог).
Текст → Router; голос → скачать → `stt` → Router с пометкой voice; документ/фото →
сохранить в inbox-папку → событие `file_received` (индексатор подхватит) + сообщение
агенту «пользователь прислал файл X»; inline-кнопки для подтверждений destructive
и действий с напоминаниями; разбиение длинных ответов, markdown-экранирование,
streaming через периодическое редактирование сообщения.

---

## 14. Local Web Chat (`channels/webchat/`) и Console (`channels/cli/`)

**Web:** FastAPI + WebSocket на `127.0.0.1`, минимальный SPA-чат: история, streaming,
кнопки подтверждений. Тот же Router — никакой своей логики.
**CLI:** REPL для разработки: диалог + служебные команды (`/tools`, `/memory`, `/reindex`,
`/jobs`) — smoke-инструмент каждого спринта.

---

## 15. MCP Client Hub и MCP Server (`mcp/`)

**Client Hub:** читает `mcp.servers` из конфига, поднимает соединения (stdio/HTTP),
получает списки тулов, оборачивает в ToolSpec (риск задаётся в конфиге для каждого
сервера/тула, по умолчанию — консервативно `destructive`), регистрирует в Registry,
переподключается при падении.
**Server:** отдельный entrypoint, экспортирует `rag_search`, `recall`, `create_task` и др.
для внешних MCP-клиентов (Claude Desktop и т.п.).

---

## 16. Config Service (`infra/config.py`)

**Ответственность:** `default.yaml` ⊕ `local.yaml` ⊕ env → валидированный Pydantic-конфиг:
пути, роли моделей, feature flags модулей, каналы, whitelisted корни ФС, allowlist хостов,
MCP-серверы, расписания, тихие часы, часовой пояс. Ошибка конфига = отказ старта
с внятным сообщением.

---

## 17. Logging / Audit (`infra/logging.py`, `infra/audit.py`)

**Ответственность:** structlog → JSON-файлы с ротацией; `request_id` через contextvars
сквозь весь путь запроса (NFR-7). Audit — отдельная append-only таблица: входящие,
tool calls (аргументы/результат/риск/подтверждение), напоминания, фоновые работы.
Инструмент `what_did_you_do` для вопросов «что ты делал вчера?».

---

## 18. Backup (`infra/backup.py`)

**Ответственность:** ночной бэкап `data/` (SQLite `VACUUM INTO`, Qdrant snapshot, конфиг)
в версионированную папку; retention 7d+4w; ежемесячный автоматический restore-тест
во временную папку со smoke-запросом; отчёт в Telegram при неудаче.

**Следующий документ:** [05-flows.md](05-flows.md).

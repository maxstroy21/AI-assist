# Этап 6. Потоки обработки

Шесть ключевых сценариев. Все остальные — комбинации этих.

---

## 1. Текстовый вопрос с поиском по документам (основной поток)

Пример: *«Найди договор с подрядчиком по экспедиции и скажи, какие там сроки»*

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant TG as Telegram Gateway
    participant R as Router
    participant A as Orchestrator
    participant CB as Context Builder
    participant M as Memory
    participant L as LLM Gateway
    participant T as Tool Registry
    participant RAG as RAG Service

    U->>TG: сообщение
    TG->>TG: whitelist check
    TG->>R: IncomingMessage
    R->>R: сессия, сохранить в messages
    R->>A: process(msg, session)
    A->>CB: build_context()
    CB->>M: preferences() + recall("договор подрядчик экспедиция")
    M-->>CB: предпочтения; факты (проект «Экспедиция», подрядчик N)
    CB-->>A: промпт (system+память+история)
    A->>L: chat(role=chat, tools=[...])
    L-->>A: tool_call: search_documents(query, filters={project})
    A->>T: execute (risk=read → сразу; audit)
    T->>RAG: search(...)
    RAG-->>T: топ-8 пассажей с цитатами
    T-->>A: ToolResult
    A->>L: chat(+результаты)
    L-->>A: финальный текст с источниками
    A->>R: OutgoingMessage (streaming)
    R->>TG: доставка (редактирование сообщения)
    TG->>U: ответ + «📄 источник: .../договор.pdf, стр. 4»
    A--)M: событие message_processed → фоновое извлечение фактов
```

---

## 2. Голосовое сообщение → задача с напоминанием

Пример: 🎤 *«Напомни мне в понедельник утром позвонить Ивану насчёт сметы»*

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant TG as Telegram Gateway
    participant L as LLM Gateway
    participant A as Orchestrator
    participant T as Tool Registry
    participant TASK as Task Manager
    participant REM as Reminder Engine
    participant S as Scheduler

    U->>TG: voice.ogg
    TG->>L: transcribe (faster-whisper)
    L-->>TG: текст
    TG->>A: IncomingMessage(text, kind=voice)  # через Router
    A->>L: chat(tools)
    L-->>A: tool_call: create_task("позвонить Ивану насчёт сметы", when="в понедельник утром")
    A->>T: execute → TASK
    TASK->>L: structured(role=extraction): "в понедельник утром" + tz + предпочтения
    L-->>TASK: {due: 2026-07-20T09:00+03}
    TASK->>REM: напоминание к задаче
    REM->>S: schedule_once(due, event=reminder_fire)
    TASK-->>A: ToolResult(task_id, due)
    A->>L: chat(+результат)
    L-->>A: «Создал задачу, напомню в пн в 9:00»
    A->>U: ответ (через Router→TG)
```

---

## 3. Срабатывание напоминания и follow-up

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant B as Event Bus
    participant REM as Reminder Engine
    participant R as Router
    actor U as Пользователь

    S->>B: reminder_fire(id)
    B->>REM: обработчик
    REM->>REM: актуально? (задача не закрыта)
    REM->>R: OutgoingMessage + кнопки [✅ Сделал] [⏰ Позже] [✖ Отмена]
    R->>U: Telegram + web-чат
    alt Пользователь: «⏰ Позже»
        U->>REM: callback (через TG→Router)
        REM->>S: schedule_once(+3h)
    else Нет реакции (условное «если я не сделал»)
        S->>B: followup_check (через 2 дня)
        B->>REM: задача всё ещё открыта → повторное напоминание
    else «✅ Сделал»
        REM->>REM: закрыть задачу, событие task_completed
    end
```

Надёжность: доставка фиксируется в `reminder_log`; при падении в момент
срабатывания APScheduler (misfire_grace) доставит после рестарта.

---

## 4. Пользователь прислал документ в Telegram

```mermaid
flowchart LR
    A["📎 документ в TG"] --> B["Gateway: сохранить в data/inbox/"]
    B --> C["событие file_received"]
    B --> D["Router → агент:<br/>«пользователь прислал файл X»"]
    C --> E["Indexer: очередь →<br/>извлечение → чанки →<br/>эмбеддинг → Qdrant"]
    D --> F["LLM: краткое резюме файла<br/>+ вопрос: куда положить?<br/>(предложение по правилам разбора)"]
    E --> G["событие document_indexed"]
    G --> H["файл доступен в поиске"]
```

---

## 5. Разрушающая файловая операция (подтверждение)

Пример: *«Разбери папку Downloads — перенеси документы по проектам и удали дубликаты»*

```mermaid
sequenceDiagram
    actor U as Пользователь
    participant A as Orchestrator
    participant T as Tool Registry
    participant F as File Ops

    U->>A: команда
    A->>T: find_duplicates (read) → список
    A->>T: move_file ×N (write) → undo-журнал, выполняется
    A->>T: delete_files (destructive!)
    T-->>A: требуется подтверждение
    A->>U: «Готов удалить 12 дубликатов: [список]. Подтверждаешь?» [Да/Нет]
    U->>A: Да
    A->>T: execute (audit: confirmed=true)
    T->>F: удаление (в корзину ОС, не мимо неё)
    A->>U: отчёт: перенесено 34, удалено 12, откат: «отмени последнюю операцию»
```

---

## 6. Фоновые циклы (ночь)

```mermaid
flowchart TD
    subgraph Ночь["Тихие часы (конфиг, напр. 02:00–06:00)"]
        S["Scheduler"] --> C1["Консолидация памяти:<br/>сессии дня → эпизоды → факты"]
        S --> C2["Полный скан источников:<br/>диффы по hash → дозаиндексация"]
        S --> C3["Бэкап data/ + retention"]
        S --> C4["Гигиена задач: просроченные →<br/>события task_overdue → follow-up"]
    end
    S2["Scheduler 08:30"] --> MB["Утренняя сводка (профиль агента<br/>morning-brief): задачи на сегодня,<br/>напоминания, вчерашние хвосты<br/>→ Telegram"]
```

**Следующий документ:** [06-roadmap.md](06-roadmap.md).

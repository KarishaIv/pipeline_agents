# Meta Agent

`meta_agent` - это LangGraph-оркестратор для аналитических вопросов по данным пайплайна. Он умеет:

- принимать вопрос пользователя;
- извлекать данные из локального Qdrant;
- считать статистики и строить графики;
- при необходимости делегировать вычисления отдельному `code_writer`;
- возвращать итоговый ответ с сохранением контекста сессии по `thread_id`.

Основной HTTP-вход находится в `src/scripts/serve_meta_agent.py`, а основной Python API - в `meta_graph_manager.invoke_graph_session(question, thread_id)`.

## Что нужно перед запуском

- Python `>=3.14`
- установленные зависимости проекта;
- доступ к Yandex LLM API;
- доступ к DVC remote в Yandex Object Storage;
- локально запущенный Qdrant.

## Установка зависимостей

Из корня репозитория:

```bash
uv sync
```

Если `uv` не используется, нужен эквивалентный install всех зависимостей из `pyproject.toml`.

## Переменные окружения

`serve_meta_agent.py` загружает переменные через `python-dotenv`, поэтому удобнее всего положить их в `.env` в корне проекта.

Пример:

```dotenv
# Yandex LLM
YANDEX_API_KEY=<your-yandex-api-key>
YANDEX_FOLDER_ID=<your-yandex-folder-id>

# DVC / Yandex Object Storage (S3-compatible)
AWS_ACCESS_KEY_ID=<your-yandex-object-storage-key-id>
AWS_SECRET_ACCESS_KEY=<your-yandex-object-storage-secret>
AWS_DEFAULT_REGION=ru-central1

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=meta-agent

# Qdrant (optional, defaults shown)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

Что важно:

- `YANDEX_API_KEY` - обязателен для всех вызовов LLM.
- `YANDEX_FOLDER_ID` - обязателен, потому что URI модели собирается в формате `gpt://<folder_id>/<model>`.
- `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` нужны для `dvc pull`, так как DVC remote настроен на S3-совместимое хранилище Yandex Cloud.
- `LANGCHAIN_*` переменные необязательны и нужны только для tracing.
- `QDRANT_HOST` и `QDRANT_PORT` можно не задавать, если Qdrant поднят локально на `localhost:6333`.

## DVC: скачать данные перед инициализацией

Мета-агент использует parquet-файлы из `data_4_qdrant/`, и они находятся под DVC:

- `data_4_qdrant/questions.parquet`
- `data_4_qdrant/personas.parquet`
- `data_4_qdrant/target_audiences.parquet`
- `data_4_qdrant/simulations.parquet`

Сначала подтяните данные из remote:

```bash
dvc pull
```

В проекте уже настроен remote `yc`:

- bucket: `s3://diplom-storage`
- endpoint: `https://storage.yandexcloud.net`

Без корректных `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` `dvc pull` не сможет скачать данные.

## Поднять Qdrant

Мета-агент ожидает доступный локальный Qdrant. Самый простой вариант:

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Директория `qdrant_storage/` игнорируется git и создается для локального состояния Qdrant.

Если Qdrant уже запущен в другом месте, выставьте `QDRANT_HOST` и `QDRANT_PORT`.

## Инициализация коллекций Qdrant

После `dvc pull` загрузите parquet-данные в Qdrant:

```bash
python -m src.scripts.init_qdrant
```

Скрипт создаст коллекции:

- `questions`
- `personas`
- `target_audiences`
- `simulations`

Если коллекции уже существуют, скрипт пересоздаст их заново.

## Запуск API

Из корня репозитория:

```bash
uvicorn src.scripts.serve_meta_agent:app --host 0.0.0.0 --port 8000
```

После старта будет доступен endpoint:

- `POST /ask` (основной, рекомендуемый)

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какие сегменты чаще соглашаются на кредит?"}'
```

Пример ответа:

```json
{
  "thread_id": "0195f3d1-...",
  "outputs": [
    {
      "type": "text",
      "text": "На основе анализа данных..."
    }
  ]
}
```

## Работа с сессиями

Параметр `thread_id` управляет историей диалога:

- `thread_id=null` или отсутствует — начать новую сессию;
- `thread_id=-1` — явно начать новую сессию;
- `thread_id=<existing-id>` — продолжить конкретную сессию.

Пример продолжения диалога:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "А теперь только по молодежной аудитории", "thread_id": "<thread-id>"}'
```

## Telegram Bot

Боты запускается как отдельный процесс, использует long polling и вызывает API `/ask`.

### Установка и конфигурация

1. Создайте Telegram бота через [@BotFather](https://t.me/botfather) и получите токен.

2. Добавьте конфигурацию в `.env`:

```dotenv
# Telegram Bot
TELEGRAM_BOT_TOKEN=<your-telegram-bot-token>
META_AGENT_API_URL=http://localhost:8000
TELEGRAM_POLL_TIMEOUT=30
TELEGRAM_REQUEST_TIMEOUT=60
TELEGRAM_SESSION_DB_PATH=data/telegram_sessions.sqlite3
TELEGRAM_THREAD_SCOPE=chat
```

### Запуск бота

Из корня репозитория (убедитесь, что API работает на http://localhost:8000):

```bash
python3 -m src.scripts.serve_telegram_bot
```

### Команды бота

- `/start` — начать диалог, показать справку
- `/help` — справка по использованию
- `/new [name]` — создать новую сессию (по умолчанию: session_1, session_2, ...)
- `/sessions` — показать список всех сессий
- `/switch <name>` — переключиться на другую сессию
- `/delete <name>` — удалить сессию

### Управление сессиями

Бот поддерживает **несколько сессий** для каждого пользователя:

- Каждая сессия — это отдельный диалог с meta-agent со своей историей
- Вы можете создавать, переключаться и удалять сессии
- Одна сессия всегда активна (помечена 🟢)
- Все сообщения идут в активную сессию

### Примеры использования

1. **Начать новый диалог:**
   ```
   /new work_analysis
   ```
   Создаст сессию с именем "work_analysis" и сделает её активной.

2. **Задать вопрос:**
   ```
   Какие сегменты клиентов чаще всего берут кредит?
   ```
   Сообщение попадёт в активную сессию, бот сохранит контекст.

3. **Создать ещё одну сессию для другой темы:**
   ```
   /new personal_finance
   ```

4. **Посмотреть все сессии:**
   ```
   /sessions
   ```
   Покажет список:
   ```
   🟢 personal_finance (updated: 2026-04-28 12:00)
   ⚪️ work_analysis (updated: 2026-04-28 11:30)
   ```

5. **Переключиться на другую сессию:**
   ```
   /switch work_analysis
   ```
   Теперь все сообщения будут в сессии "work_analysis".

6. **Удалить ненужную сессию:**
   ```
   /delete old_session
   ```
   Нельзя удалить активную сессию — сначала переключитесь на другую.

### Архитектура

- Бот использует **long polling** (не требует webhook).
- **Множественные сессии**: каждый пользователь может иметь несколько параллельных диалогов (сессий).
- Каждая сессия имеет уникальное имя и привязана к своему `thread_id` в meta-agent.
- Одна сессия всегда активна — все сообщения идут в неё.
- Все сообщения обрабатываются по очереди (per-chat lock).
- Длинные ответы автоматически разбиваются на несколько сообщений.
- Будущие расширения поддерживают JSON и файловые выходы (PDF, графики).



## Что находится внутри `meta_agent`

- `graph.py` - сборка и запуск LangGraph-графа;
- `nodes.py` - узлы `supervisor`, `data_extractor`, `analyzer`, `code_writer`;
- `agent_factory.py` - создание LLM-агентов поверх Yandex API;
- `tools/qdrant_tools.py` - поиск, фильтрация, скролл и retrieve по Qdrant;
- `tools/analyzer_tools.py` - статистики, суммаризация и построение графиков;
- `services/qdrant.py` - singleton-клиент к Qdrant;
- `catalog.py` - список доступных коллекций и их описания.

## Типовой порядок запуска с нуля

```bash
uv sync
dvc pull
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
python -m src.scripts.init_qdrant
uvicorn src.scripts.serve_meta_agent:app --host 0.0.0.0 --port 8000
```

## Частые проблемы

`dvc pull` не работает:
- проверьте `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`;
- проверьте, что есть доступ к Yandex Object Storage.

LLM-вызовы падают:
- проверьте `YANDEX_API_KEY`;
- проверьте `YANDEX_FOLDER_ID`;
- проверьте, что ключ имеет доступ к нужной модели.

Qdrant-инструменты возвращают ошибки:
- проверьте, что Qdrant запущен;
- проверьте, что VPN выключен;
- проверьте `QDRANT_HOST` и `QDRANT_PORT`;
- убедитесь, что был выполнен `python -m src.scripts.init_qdrant`.

Нет данных в коллекциях:
- проверьте, что parquet-файлы были скачаны через `dvc pull`;
- убедитесь, что в `data_4_qdrant/` есть все четыре файла.

## Testing

All test logic is located in `src/meta_agent/test/` (per project requirements).

The project includes a `Makefile` with convenient commands. The pytest configuration (`pythonpath = [".", "src"]`) has been aligned so that imports of the form `from src.meta_agent...` work when running from the project root.

Run tests with:

```bash
# Install test deps (once)
uv sync --extra test

# Preferred: run meta_agent tests
make test-meta
# or
make test

# With coverage report
make test-cov

# Specific modules
uv run pytest src/meta_agent/test/test_utils.py -q
uv run pytest src/meta_agent/test/tools/ -q
```

See `Makefile` for `test`, `test-cov`, `lint`, `format` targets.

- **Coverage target**: >80% for utils, tools, nodes, graph.
- Uses `pytest`, `pytest-asyncio`, `pytest-mock`.
- Heavy mocking of LLM (`run_agent`), QdrantService, pandas/matplotlib, OpenAI client.
- Tests cover pure functions (history summarization/compression, state reducers, routing), all tools (DTO, Qdrant, analyzer, code execution sandbox, decision tools), nodes, graph construction/topology, and config/prompts/catalog.
- Fixtures in `conftest.py` provide `meta_state`, `mock_qdrant_service`, `mock_run_agent`, `temp_charts_dir`, etc.
- No live LLM or Qdrant calls in unit tests.

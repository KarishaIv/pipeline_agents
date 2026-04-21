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

- `GET /ask`

Пример запроса:

```bash
curl "http://127.0.0.1:8000/ask?q=Какие сегменты чаще соглашаются на кредит?"
```

Пример ответа:

```json
{
  "answer": "...",
  "thread_id": "0195f3d1-...."
}
```

## Работа с сессиями

Параметр `thread_id` управляет историей диалога:

- `thread_id=-1` - начать новую сессию;
- `thread_id=<existing-id>` - продолжить конкретную сессию;
- без `thread_id` - будет создан новый идентификатор на стороне сервиса.

Пример продолжения диалога:

```bash
curl "http://127.0.0.1:8000/ask?q=А теперь только по молодежной аудитории&thread_id=<thread-id>"
```

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

Run tests with:

```bash
# Install test deps
uv sync --extra test

# Run full test suite with coverage
uv run pytest src/meta_agent/test -q --cov=src/meta_agent --cov-report=term-missing

# Or specific modules
uv run pytest src/meta_agent/test/test_utils.py -q
uv run pytest src/meta_agent/test/tools/ -q
```

- **Coverage target**: >80% for utils, tools, nodes, graph.
- Uses `pytest`, `pytest-asyncio`, `pytest-mock`.
- Heavy mocking of LLM (`run_agent`), QdrantService, pandas/matplotlib, OpenAI client.
- Tests cover pure functions (history truncation, state reducers, routing), all tools (DTO, Qdrant, analyzer, code execution sandbox, decision tools), nodes, graph construction/topology, and config/prompts/catalog.
- Fixtures in `conftest.py` provide `meta_state`, `mock_qdrant_service`, `mock_run_agent`, `temp_charts_dir`, etc.
- No live LLM or Qdrant calls in unit tests.

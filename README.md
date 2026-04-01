# Pipeline agents

Репозиторий объединяет два направления:

1. **Симуляция опросов и мультиагентная модель** — PGM, перенос OCEAN, сценарии на Yandex GPT (`main.py`, каталог `src/`).
2. **RAG по новостям Telegram** — парсинг каналов, подготовка датасета, эмбеддинги E5, векторное хранилище **Qdrant**, семантический поиск (`rag/`, `tg_parser/`).

Актуальная схема и команды ниже ориентированы на рабочую ветку **`prepare_data`**.

---

## Установка

```bash
cd pipeline_agents
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Запуск скриптов из каталога **`pipeline_agents`** (корень репозитория), чтобы работали импорты вида `from rag....`.

---

## Переменные окружения

| Назначение | Переменные |
|------------|------------|
| Hugging Face (скачивание моделей) | `HF_TOKEN` или `HUGGING_FACE_HUB_TOKEN` (можно задать через `config.set_hf_token(...)` после импорта `config`) |
| Telegram (парсер) | `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`; опционально `TELEGRAM_SESSION` (имя файла сессии) |
| Эмбеддинги RAG | `RAG_EMBEDDING_MODEL` — id модели на HF (по умолчанию в коде ориентир на `intfloat/multilingual-e5-large`) |
| Бэкенд эмбеддингов | `RAG_EMBEDDING_BACKEND` — `auto` \| `onnx` \| `sentence_transformers`. В режиме `auto` используется ONNX, если в каталоге модели есть экспортированные `model.onnx` / `model.int8.onnx`, иначе — `sentence-transformers` |
| Каталог ONNX | `RAG_ONNX_MODEL_DIR` — папка с `model.onnx` или `model.int8.onnx` и файлами токенизатора; если не задана — используется кэш `.cache/onnx_bench/<модель>/` |
| INT8 ONNX | `RAG_ONNX_USE_INT8` — `1` (по умолчанию) предпочитать `model.int8.onnx`, `0` — только FP32 |
| Длина токенизации | `RAG_EMBEDDING_MAX_LENGTH` (по умолчанию `512`) |

Симуляция опросов: ключ Yandex GPT передаётся в **`main.py`** как `--api_key` (и при необходимости `--folder_id`).

---

## RAG: поток данных

```text
Telegram (JSON в папке)  →  prep_rag  →  rag_docs.parquet
                                    ↓
                            ingest_rag (эмбеддинги + upsert в Qdrant)
                                    ↓
                            search_rag (запрос + фильтры по дате/агенту)
```

### 1. Парсинг каналов

Скрипт: `tg_parser/more_posts.py`.

Перед запуском задайте `TELEGRAM_API_ID` и `TELEGRAM_API_HASH`.

Примеры:

```bash
# Посты за «вчера» (или за день --day YYYY-MM-DD), папка с префиксом по умолчанию
python tg_parser/more_posts.py

# Несколько дней назад (режим hours + days-back)
python tg_parser/more_posts.py --mode hours --days-back 30
```

Параллельность и паузы: `--max-concurrent`, `--batch-sleep`. Вывод — JSON-файлы по каналам в создаваемой папке (имя зависит от режима и префикса `--output-prefix`).

### 2. Подготовка датасета для RAG

Скрипт: `rag/prep_rag.py`. Собирает посты из папки с дампами, фильтрует явную рекламу, подрезает типичные промо-хвосты, проставляет категории и **агента**, сохраняет **`rag_docs.parquet`** (по умолчанию).

В точке входа `__main__` зашита папка-источник `telegram_data_last_day` — при другом имени каталога вызовите из Python:

```python
from rag.prep_rag import prepare_telegram_rag
prepare_telegram_rag("ваша_папка_с_json", output_parquet="rag_docs.parquet")
```

или временно измените аргумент в конце `rag/prep_rag.py`.

### 3. Загрузка в Qdrant

Скрипт: `rag/ingest_rag.py`. Читает `.parquet` или `.jsonl`, считает эмбеддинги через `rag/e5_embeddings.py`, делает **upsert** по стабильному `doc_id`. Локальная БД — папка на диске (без Docker), по умолчанию `qdrant_data/`.

```bash
python rag/ingest_rag.py --local --input rag_docs.parquet --collection telegram_news_e5_large
```

Полезные флаги:

- `--recreate` — пересоздать коллекцию (удалить и создать заново).
- `--prune-older-than-days 60` — после загрузки удалить точки с датой в payload старше N дней (`0` — не резать).

Одна и та же коллекция дополняется новыми днями через повторный ingest без `--recreate` (дубликаты с тем же `doc_id` перезаписываются upsert’ом).

### 4. Поиск

Скрипт: `rag/search_rag.py`. Семантический поиск; по умолчанию **окно 14 дней** (`--window-days`) и **`--top-k` 15**; опционально фильтр **`--agent`** и бонус за свежесть (`--prefer-recent` / `--no-prefer-recent`, `--recency-weight`).

```bash
python rag/search_rag.py "льготная ипотека ставки" --local --collection telegram_news_e5_large --agent real_estate --top-k 8
python rag/search_rag.py "курс доллара" --local --window-days 30 --agent currency
```

Значения **`agent`** должны совпадать с тем, что записал `prep_rag` в метаданные: `macroeconomy`, `banks`, `currency`, `real_estate`, `social_news` (см. `AGENT_BY_CATEGORY` и исключение для валютных каналов в `rag/prep_rag.py`).

---

## ONNX (ускорение опционально)

Экспорт и бенч PyTorch / ONNX / INT8:

```bash
python -m rag.onnx_quant_bench_e5 --model intfloat/multilingual-e5-large --iters 20 --batches 1 8 32
```

Артефакты попадают в `.cache/onnx_bench/<модель>/`. После этого при `RAG_EMBEDDING_BACKEND=auto` ingest и search смогут использовать ONNX, если модель и путь совпадают с `RAG_EMBEDDING_MODEL` и при необходимости `RAG_ONNX_MODEL_DIR`.

---

## Сравнение моделей эмбеддингов

Если для `e5-small` / `e5-base` / `e5-large` заведены **разные коллекции** Qdrant, можно сравнить качество и время поиска:

```bash
python rag/compare_models.py --local --window-days 14 --top-k 5
```

У `compare_models.py` свои дефолты: например, **`--window-days` по умолчанию 30** (у `search_rag` — 14), **`--top-k` по умолчанию 8**. Отчёты пишутся в `outputs/` (или в `--out-dir`).

---

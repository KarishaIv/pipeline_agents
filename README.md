# Survey Pipeline with Structured Reasoning

Репозиторий для генерации синтетических персон и запуска `survey pipeline (опросного пайплайна)`.

Текущий основной инженерный результат в этой ветке:
- `structured survey mode (структурированный survey-режим)`;
- поддержка внешнего `news context (новостного контекста)` через `JSON snapshot (JSON-снимок)`;
- `survey benchmark (survey-бенчмарк)` для сравнения `legacy (старого режима)` и `structured (структурированного режима)`.

## Ключевые файлы

- `main.py` — главный `CLI entrypoint (интерфейс командной строки)`.
- `src/orchestration.py` — сборка полного пайплайна.
- `src/core/simulation_manager.py` — запуск симуляций и выбор `survey mode (survey-режима)`.
- `src/agents/survey_agent.py` — `legacy survey runtime (старый survey-рантайм)`.
- `src/agents/structured_survey_reasoner.py` — `structured survey runtime (структурированный survey-рантайм)`.
- `src/agents/survey_news_adapter.py` — адаптация `news context (новостного контекста)` к вопросам опроса.
- `src/schemas/news_context_schema.py` — схема и `compatibility normalization (нормализация совместимости)` входных `news JSON (news JSON-файлов)`.
- `scripts/benchmarks/benchmark_survey_reasoning.py` — `survey benchmark (survey-бенчмарк)`.

## Окружение

Рекомендуемая среда:
- `Python 3.12`
- `.env` с:
  - `YANDEX_API_KEY` — предпочтительно;
  - `YANDEX_FOLDER_ID`.

Для обратной совместимости код также принимает `OPENAI_API_KEY` как имя переменной для ключа, но в общей ветке лучше использовать `YANDEX_API_KEY`.

Установка:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

## Данные через DVC

`main (главная ветка)` уже перевела часть данных в `DVC (систему версионирования данных)`, поэтому перед запуском нужно либо иметь локальные файлы, либо подтянуть их:

```bash
dvc pull data/evidence.json.dvc data/survey_questions.json.dvc data/Nemotron.dvc data/Synthetic.dvc
```

Если планируется работа с дополнительными датасетами, можно подтянуть и остальные `.dvc`-файлы из `data/` и `data_4_qdrant/`.

## Быстрый запуск

Рекомендуемый режим по умолчанию — `structured survey mode (структурированный survey-режим)`.

```bash
source .venv312/bin/activate
python main.py \
  --agent_mode survey \
  --survey-mode structured \
  --evidence data/evidence.json \
  --output outputs/ \
  --concurrency 15 \
  --timeout 60
```

Этот запуск:
- загружает `evidence (описание аудиторий)`;
- генерирует или фильтрует персоны;
- загружает вопросы из `data/survey_questions.json`;
- прогоняет для каждой персоны `structured survey reasoner (структурированный survey-рантайм)`;
- сохраняет профили и агрегированные результаты в `outputs/`.

## Режимы survey

### Structured

```bash
python main.py --agent_mode survey --survey-mode structured
```

Это основной режим. Он использует:
- явные `voices (голоса)`;
- кодовую агрегацию;
- `resolver (разрешитель конфликта)`;
- поддержку `news context (новостного контекста)`.

### Legacy

```bash
python main.py --agent_mode survey --survey-mode legacy
```

Этот режим нужен в основном как `baseline (базовый режим)` для сравнения с `structured (структурированным режимом)`.

## Подключение news context

Для `structured survey mode (структурированного survey-режима)` можно передать внешний `news context (новостной контекст)` через `--news-context-path`.

В репозитории уже лежат два готовых `news snapshots (снимка новостного контекста)` под текущие рабочие аудитории:
- `data/news_context/context_mothers_35_39_20260426.json`
- `data/news_context/context_fathers_45_49_20260426.json`

Пример запуска:

```bash
python main.py \
  --agent_mode survey \
  --survey-mode structured \
  --news-context-path data/news_context/context_mothers_35_39_20260426.json
```

Важно:
- `main.py` принимает только один `news context file (файл новостного контекста)` на запуск;
- если в наборе персон смешаны `mothers (матери)` и `fathers (отцы)`, один контекст будет точнее для одной аудитории и слабее для другой;
- для строгой оценки влияния новостей лучше запускать `benchmark (бенчмарк)` отдельно по аудиториям.

## Основные флаги

- `--agent_mode survey|credit`
  - общий режим пайплайна; в этой ветке основной рабочий путь — `survey`.
- `--survey-mode legacy|structured`
  - выбор между старым и новым `survey runtime (survey-рантаймом)`.
- `--news-context-path PATH`
  - путь к входному `news JSON (news JSON-файлу)`.
- `--evidence PATH`
  - путь к `evidence JSON (evidence JSON-файлу)`.
- `--output PATH`
  - директория сохранения результатов.
- `--concurrency N`
  - число параллельных симуляций персон.
- `--timeout SECONDS`
  - таймаут на одну персону.
- `--no-pgm`
  - отключает `PGM generation (PGM-генерацию)`.
- `--no-oceanflag`
  - отключает расчет `OCEAN traits (личностных черт OCEAN)`.

## Survey benchmark

Для воспроизводимого сравнения `legacy vs structured (старого и нового режима)` и `with news vs without news (с новостями и без новостей)`:

```bash
source .venv312/bin/activate
python scripts/benchmarks/benchmark_survey_reasoning.py \
  --profiles-glob "outputs/profile_*.json" \
  --profile-sample 10 \
  --question-sample 8 \
  --repeats 2 \
  --concurrency 1 \
  --survey-modes structured \
  --judge-sample 16 \
  --locale ru \
  --seed 17 \
  --judge-seed 23 \
  --out-dir outputs/benchmarks/survey_reasoning/example_structured_run
```

Пример с новостным контекстом:

```bash
python scripts/benchmarks/benchmark_survey_reasoning.py \
  --profiles-glob "outputs/profile_*.json" \
  --profile-sample 10 \
  --question-sample 8 \
  --repeats 2 \
  --concurrency 1 \
  --survey-modes structured \
  --judge-sample 16 \
  --locale ru \
  --seed 17 \
  --judge-seed 23 \
  --news-context-path data/news_context/context_mothers_35_39_20260426.json \
  --out-dir outputs/benchmarks/survey_reasoning/example_structured_news_run
```

Главные выходы `survey benchmark (survey-бенчмарка)`:
- `metrics.json`
- `predictions.csv`
- `judge_results.csv`
- `manifest.json`

## Что не входит в основной пайплайн

Внешний генератор `news context (новостного контекста)` из папки `multi_agent_rag/` не является частью основного пайплайна этого репозитория. В эту ветку перенесена только:
- совместимость с его `JSON outputs (JSON-выходами)`;
- поддержка готовых `news snapshots (снимков новостного контекста)`.

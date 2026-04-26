# Multi-Agent Persona and Survey Pipeline

Репозиторий для генерации синтетических персон и запуска мультиагентных симуляций в двух режимах:
- `credit mode (кредитный режим)` для моделирования реакции на кредитное предложение;
- `survey mode (опросный режим)` для генерации ответов на опросные вопросы.

Текущий основной инженерный результат в репозитории — `structured survey mode (структурированный survey-режим)`. Он работает через явные `voices (голоса)`, кодовую агрегацию и поддерживает подключение `news context (новостного контекста)`.

## Что есть в репозитории

- `main.py` — главный `CLI entrypoint (интерфейс командной строки)`.
- `src/orchestration.py` — сборка полного пайплайна.
- `src/core/simulation_manager.py` — маршрутизация в `credit` или `survey` режим.
- `src/agents/structured_survey_reasoner.py` — новый `structured survey runtime (структурированный survey-рантайм)`.
- `src/agents/survey_news_adapter.py` — `news adapter (адаптер новостного контекста)` для survey.
- `src/schemas/news_context_schema.py` — схема и `compatibility normalization (нормализация совместимости)` входных news JSON.
- `scripts/benchmarks/benchmark_survey_reasoning.py` — основной `survey benchmark (survey-бенчмарк)`.

## Зависимости и окружение

Рекомендуемая среда:
- `Python 3.12`
- `.env` с переменными:
  - `OPENAI_API_KEY`
  - `YANDEX_FOLDER_ID`

Установка:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

## Быстрый запуск survey-пайплайна

### Рекомендуемый режим: structured survey

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

Что делает этот запуск:
- загружает `evidence (описание аудиторий)` из `data/evidence.json`;
- генерирует или фильтрует персоны;
- загружает вопросы из `data/survey_questions.json`;
- прогоняет для каждой персоны `structured survey reasoner (структурированный survey-рантайм)`;
- сохраняет `profile_*.json`, `survey_summary.json` и остальные артефакты в `outputs/`.

### Старый режим: legacy survey

```bash
source .venv312/bin/activate
python main.py \
  --agent_mode survey \
  --survey-mode legacy
```

Этот режим нужен в основном для сравнения с `structured survey mode (структурированным survey-режимом)`.

## Подключение news context

Для `structured survey mode (структурированного survey-режима)` можно передать внешний `news context (новостной контекст)` через `--news-context-path`.

В репозитории уже лежат два готовых `news snapshots (снимка новостного контекста)` под текущие рабочие аудитории:
- `data/news_context/context_mothers_35_39_20260426.json`
- `data/news_context/context_fathers_45_49_20260426.json`

Пример запуска:

```bash
source .venv312/bin/activate
python main.py \
  --agent_mode survey \
  --survey-mode structured \
  --news-context-path data/news_context/context_mothers_35_39_20260426.json
```

Важно:
- `main.py` принимает только один `news context file (файл новостного контекста)` на запуск;
- если в наборе персон смешаны `mothers` и `fathers`, то один и тот же контекст будет точнее для одной аудитории и слабее для другой;
- для строгой оценки эффекта новостей лучше запускать `survey benchmark (survey-бенчмарк)` отдельно по аудиториям.

## Основные флаги

- `--agent_mode credit|survey`
  - выбирает общий режим пайплайна.
- `--survey-mode legacy|structured`
  - выбирает старый или новый survey runtime.
- `--decision-mode direct|compact_debate`
  - режим кредитного рассуждения для `credit mode (кредитного режима)`.
- `--news-context-path PATH`
  - путь к входному `news context JSON (news context JSON-файлу)` для `structured survey` или `compact_debate`.
- `--evidence PATH`
  - путь к `evidence JSON (evidence JSON-файлу)` с описанием целевых аудиторий.
- `--output PATH`
  - директория сохранения результатов.
- `--concurrency N`
  - число параллельных симуляций персон.
- `--timeout SECONDS`
  - таймаут на одну персону.
- `--no-pgm`
  - использовать не PGM-генерацию, а фильтрацию реальных данных.
- `--no-oceanflag`
  - отключить перенос `OCEAN traits (личностных черт OCEAN)`.

## Survey benchmark

Для воспроизводимого сравнения `legacy vs structured (старого и нового режима)` и `with news vs without news (с новостями и без новостей)` используй:

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
  --news-context-path data/news_context/context_mothers_35_39_20260426.json \
  --out-dir outputs/benchmarks/survey_reasoning/example_structured_news_run
```

Главные выходы `survey benchmark (survey-бенчмарка)`:
- `metrics.json`
- `predictions.csv`
- `judge_results.csv`
- `manifest.json`

## Credit mode

`credit mode (кредитный режим)` остается доступным:

```bash
source .venv312/bin/activate
python main.py \
  --agent_mode credit \
  --decision-mode compact_debate
```

При необходимости туда тоже можно передать `--news-context-path`, но основной финальный результат в репозитории сейчас связан именно с `structured survey mode (структурированным survey-режимом)`.

## Что не входит в основной merge

Внешний генератор `news context (новостного контекста)` из папки `multi_agent_rag/` не является частью основного пайплайна этого репозитория. В основной код здесь перенесена только совместимость с его `JSON outputs (JSON-выходами)` и примеры готовых `news snapshots (снимков новостного контекста)` для текущих аудиторий.

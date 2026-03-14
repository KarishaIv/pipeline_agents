# Survey Pipeline with Multi-Agent Simulation

Комплексная система для моделирования поведения российских клиентов банка при принятии решений о кредите. Использует Probabilistic Graphical Models (PGM) для генерации синтетических персон, k-NN matching для переноса личностных черт из американского датасета (Nemotron), и мультиагентную симуляцию на базе LLM для моделирования эмоций, поведения и финансовых решений.

---

## 📊 Схема пайплайна

![Схема пайплайна](diagram.png)

---

## 🚀 Запуск

### Требования
- Python 3.10+
- OpenAI API ключ
- Установленные зависимости (см. `requirements.txt`)

### Базовый запуск
```bash
python main.py --api_key YOUR_OPENAI_API_KEY
```

### Полный запуск с параметрами
```bash
python main.py \
  --api_key YOUR_OPENAI_API_KEY \
  --evidence data/evidence.json \
  --synthetic_size 100 \
  --nemo_size 5000 \
  --output outputs/ \
  --simulation_steps 3 \
  --concurrency 4 \
  --timeout 180.0
```

---

## ⚙️ Гиперпараметры и конфигурация

### Параметры командной строки

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|-------------|----------|
| `--api_key` | str | **обязательный** | OpenAI API ключ для LLM вызовов |
| `--evidence` | str | `data/evidence.json` | Путь к JSON с evidence-данными для PGM, сам запрос на социальную группу для исследования |
| `--synthetic_size` | int | 10 | Количество синтетических российских персон |
| `--nemo_size` | int | 5000 | Размер американского датасета (Nemotron) |
| `--output` | str | `outputs/` | Директория для сохранения результатов |
| `--simulation_steps` | int | 2 | Количество шагов в мультиагентной симуляции |
| `--concurrency` | int | 3 | Число параллельных симуляций персон |
| `--timeout` | float | 180.0 | Таймаут на одну персону (секунды) |

### Конфигурация в `config.py`

```python
# K-NN параметры
TOP_N_CATEGORIES = 10      # Топ-N категориальных комбинаций
TOP_N_NEIGHBORS = 10       # Топ-N ближайших американцев

# Размеры датасетов
DEFAULT_SYNTHETIC_SIZE = 10
DEFAULT_NEMO_SIZE = 5000

# Пути к данным
DATA_PATHS = {
    'evidence': 'data/evidence.json',
    'synthetic_data': 'data/synthetic_personas.csv',
    'nemotron_data': 'data/nemotron_americans.csv'
}
```

---

## 📖 Общая идея этапов

### Этап 1-4: Генерация синтетических российских персон (PGM)

**Цель**: Создать реалистичных российских клиентов с известными демографическими параметрами.

**Процесс**:
1. **Загрузка данных**: Считываем evidence (условия запроса, например, "мужчина 30-40 лет, Москва") и базовый датасет (реальные транзакционные данные или синтетические)
2. **Preprocessing**: Дискретизируем continuous переменные (возраст, доход → группы)
3. **Обучение PGM**: Строим Discrete Bayesian Network с каузальными связями:
   ```
   age_group → marital_status
   age_group → children_group
   education → occupation → income_level
   region_type → income_level
   ...
   ```
4. **Генерация**: Используем likelihood-weighted sampling с evidence, чтобы сгенерировать N российских персон, удовлетворяющих запросу evidence

**Результат**: DataFrame с синтетическими российскими персонами (демография без OCEAN)

---

### Этап 5-8: Transfer OCEAN из американского датасета

**Цель**: Назначить каждой российской персоне личностные черты (Big Five OCEAN).

**Проблема**: У нас нет данных OCEAN для россиян, но есть для американцев (Nemotron).

**Решение через k-NN**:

1. **Category Matching**: Сначала находим американцев с похожими категориальными признаками (пол, образование, профессия - все что есть по соц.демографии)
   - Вычисляем Euclidean distance по категориям
   - Отбираем топ-10 комбинаций
   - Создаём filtered pool американцев

2. **k-NN**: Для каждой российской персоны:
   - Вычисляем расстояние до всех американцев из filtered pool
   - Находим k=10 ближайших соседей
   - Извлекаем их OCEAN профили через агента

3. **Агрегация OCEAN**:
   - Вычисляем mean и std по OCEAN соседей
   - Добавляем к профилю российской персоны как `openness_mean`, `openness_std`, etc.

**Результат**: Российские персоны с демографией + OCEAN статистикой

---

### Этап 9: Multi-Agent Simulation (Ядро системы)

**Цель**: Смоделировать взаимодействие клиента с банковским приложением и принятие решения о кредите.

**Архитектура мультиагентной системы**:

```
MultiAgentSystem (оркестратор)
  │
  ├── PersonaAgent (клиент)
  │     ├── EmotionAgent (эмоции)
  │     └── ToolAgent (ответы приложения)
  │
  ├── FinancialAgent (банк)
  │
  └── DecisionAgent (финальное решение)
```

#### 9.1 PersonaAgent — Моделирование клиента

**Инициализация**:
- Определяет цель визита в приложение (через LLM): "оформить кредит", "проверить баланс", "оплатить счета", etc.
- EmotionAgent устанавливает начальное эмоциональное состояние (mood, stress, confidence, bank_trust, urgency)

**Цикл симуляции (N шагов)**:
Каждый шаг = одно действие клиента в приложении:

1. **PersonaAgent.act_step()**:
   - LLM генерирует следующее действие на основе:
     - Цели клиента
     - Текущих эмоций
     - Истории последних действий
   - Пример: "нажал на раздел Кредиты", "открыл калькулятор кредита", "запросил условия"

2. **ToolAgent.respond()**:
   - Симулирует ответ банковского приложения
   - Примеры: "показал список кредитов", "одобрил предварительную заявку", "вернул ошибку"

3. **EmotionAgent.update_state()**:
   - Обновляет эмоции после действия и ответа
   - Если одобрение → stress↓, confidence↑
   - Если отказ → stress↑, confidence↓, bank_trust↓

4. **История обновляется**: Сохраняется запись (action, tool_response, emotional_state)

**Результат цикла**: История взаимодействий + финальное эмоциональное состояние

#### 9.2 FinancialAgent — Генерация push-уведомления

После N шагов симуляции:

1. **Создание персонализированного push** (через LLM):
   - Анализирует профиль и поведение клиента
   - Генерирует уведомление: "Специальное предложение: кредит под 12% для вас!"

2. **Предварительная оценка** (через LLM):
   - Предсказывает вероятность согласия клиента
   - Объясняет reasoning

**Результат**: Push-уведомление + prediction вероятности

#### 9.3 PersonaAgent.react_to_push()

Клиент получает push:
- LLM моделирует реакцию: "заинтересуется", "проигнорирует", "раздражится"
- EmotionAgent обновляет эмоции после push

#### 9.4 DecisionAgent — Финальное решение

**Вход**:
- Полный профиль клиента
- История всех действий в приложении
- Финальное эмоциональное состояние
- Push-уведомление и реакция

**Процесс** (через LLM):
- Комплексный анализ всех факторов
- Принятие решения от лица клиента: брать ли кредит
- Учёт целей, эмоций, финансового благополучия

**Результат**: 
```json
{
  "will_take_credit": true/false,
  "decision_reasoning": "объяснение",
  "emotional_factors": ["стресс снизился", "высокая уверенность"],
  "probability_score": 0.75
}
```

---

### Этап 10: Сохранение результатов

**Асинхронное потоковое сохранение**:
- Каждая персона сохраняется сразу после завершения симуляции
- Не ждём окончания всех персон
- Минимизация потери данных при сбоях

**Структура выходных данных**:
```
outputs/
  └── sim_20231105_143022/
      ├── persona_0_full.json      # Полная история симуляции
      ├── persona_1_full.json
      ├── ...
      ├── summary_20231105_143022.json  # Агрегированная статистика
      └── visualizations/
          ├── persona_0_emotions.png    # График эмоций
          └── ...
```

**Сохраняемые данные для каждой персоны**:
- Исходный профиль (демография + OCEAN)
- Цель визита
- Полная история шагов (actions, tool responses, emotions)
- Push-уведомление и реакция
- Финальное решение с обоснованием
- Визуализация динамики эмоций (опционально)

---

## 🔧 Параллелизация и производительность

### Асинхронная архитектура

**SimulationManager** управляет параллельным запуском симуляций:

```python
# Создание менеджера
manager = SimulationManager(
    concurrency=4,      # Одновременно 4 персоны
    timeout=180.0,      # Таймаут на персону
    run_retries=1       # Повторные попытки при сбое
)

# Асинхронный запуск
results = await manager.run_many(personas, steps=3)
```

**Механизм ограничения параллелизма**:
- `asyncio.Semaphore(concurrency)` — не более N персон одновременно
- `asyncio.wait_for(timeout)` — принудительное прерывание зависших симуляций
- `asyncio.as_completed()` — обработка результатов по мере готовности

---

## 📚 Ключевые компоненты

### Agents

| Agent | Роль | Технологии |
|-------|------|-----------|
| **PersonaAgent** | Моделирование клиента | LLM (GPT), structured outputs |
| **EmotionAgent** | Отслеживание эмоций | LLM psychological prompts |
| **ToolAgent** | Симуляция приложения | Rule-based + LLM |
| **FinancialAgent** | Генерация push | LLM personalization |
| **DecisionAgent** | Принятие решения | LLM reasoning |

### Schemas (Pydantic)

Все данные валидируются через Pydantic:
- `PersonaGoal`, `PersonaAction`, `PersonaReaction`, `PersonaSessionRecord`
- `EmotionalStateSchema` (mood, stress, confidence, bank_trust, urgency)
- `ToolResponseSchema` (status, message, data)
- `FinancialPush`, `FinancialPrediction`
- `DecisionOutcome` (will_take_credit, reasoning, emotional_factors)

### Core Utilities

- **llm_utils.py**: `robust_llm_call()` с retry logic и structured outputs
- **storage.py**: Асинхронное сохранение JSON
- **visualization.py**: Графики динамики эмоций (matplotlib)

---

## 🔬 Технические детали

### PGM (Probabilistic Graphical Model)

**Библиотека**: pgmpy  
**Тип модели**: Discrete Bayesian Network  
**Обучение**: Maximum Likelihood Estimation  
**Inference**: Likelihood-weighted sampling

**Граф зависимостей**:
```
age_group → marital_status, children_group, education, income_level
education → occupation, income_level
marital_status → children_group
region_type → income_level
occupation → income_level
gender → marital_status, income_level
```

### k-NN Matching

**Библиотека**: scipy.spatial.distance  
**Метрика**: Euclidean distance (после нормализации)  
**Процесс**:
1. Category filtering (categorical exact match)
2. Distance computation (continuous features)
3. Top-k selection
4. OCEAN aggregation (mean + std)

### LLM Integration

**Модель**: OpenAI GPT (gpt-4)  
**Режим**: Structured outputs (JSON mode)  
**Retry**: До 3 попыток при ошибках API  
**Timeout**: Конфигурируемый per-call

---
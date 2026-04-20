"""Конфигурационные константы мета-агента.

Все значения могут быть переопределены через переменные окружения.
"""

import os
from pathlib import Path

from config import PROJECT_ROOT

# Лимиты итераций
MAX_SUPERVISOR_ITERATIONS = 10
MAX_AGENT_ITERATIONS = 20
MAX_HISTORY_CHARS = 12_000
MAX_DELEGATED_ATTEMPTS = 6

# Конфигурация Qdrant (переопределяется через env)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Пути и таймауты
CHARTS_DIR: Path = PROJECT_ROOT / "charts"
CODE_TIMEOUT = 30

# Модели LLM (BIG_MODEL используется для code_writer)
LLM_MODEL = "yandexgpt-5.1"
BIG_MODEL = "yandexgpt-5.1"
# BIG_MODEL = "qwen3-235b-a22b-fp8"
